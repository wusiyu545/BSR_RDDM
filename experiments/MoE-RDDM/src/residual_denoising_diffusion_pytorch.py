import copy
import glob
import math
import os
import random
import time
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import Augmentor
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from my_datasets.get_dataset import dataset
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
from torch import einsum, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm
# 画图与 SDT 路由
from torch.utils.tensorboard import SummaryWriter
from .sdt_module import SoftDecisionTree, get_sdt_probs  # ✅ 补全 "." 建立精确映射

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
# helpers functions


def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class SDTFeatureRouter(nn.Module):
    def __init__(self, dim, num_experts=4, init_sdt_scale=2.0, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_experts)
        )

        # =======================================================
        # 🚀 核心升级：可学习的动态尺度参数 (Learnable Scale)
        # 初始化为较高的 2.0，确保训练初期 SDT 拥有绝对指挥权。
        # 随着训练推进，网络会自动为不同的 U-Net 深度层学出最完美的平衡比例！
        # =======================================================
        self.sdt_scale = nn.Parameter(torch.tensor(init_sdt_scale))

    def forward(self, x, sdt_probs=None):
        """
        x: [B, C, H, W]
        sdt_probs: [B, E] or None
        """
        feat = self.pool(x).flatten(1)  # [B, C]
        feat_logits = self.fc(feat)  # [B, E]

        feat_logits = feat_logits - feat_logits.mean(dim=-1, keepdim=True)

        if sdt_probs is not None:
            sdt_logits = torch.log(sdt_probs.clamp_min(1e-8))

            # 🛡️ 保护机制：使用 F.relu 确保 scale 永远是非负数
            # 防止网络走火入魔学出负数，导致 SDT 逻辑完全反转
            actual_scale = F.relu(self.sdt_scale)

            logits = feat_logits + actual_scale * sdt_logits
        else:
            logits = feat_logits

        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights, logits, feat_logits

# =========================================================================
# 🚀 进阶架构：特征级混合专家残差块 (Feature-level MoE ResnetBlock)
# =========================================================================
class MoEResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        groups=8,
        num_experts=4,
        expert_depth=1,
        use_shared=False,  # 默认关闭并联共享，因为我们在 U-Net 外部已经串联了共享块
        init_sdt_scale=2.0,
        temperature=1.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_depth = expert_depth
        self.use_shared = use_shared

        # 🚀 1. 挂载自适应特征路由器 (融合 SDT 先验)
        self.router = SDTFeatureRouter(
            dim=dim,
            num_experts=num_experts,
            init_sdt_scale=init_sdt_scale,
            temperature=temperature
        )

        # 🌟 2. 可选的并联共享分支 (原生时间感知块)
        if self.use_shared:
            self.shared_block = ResnetBlock(dim, dim_out, time_emb_dim=time_emb_dim, groups=groups)

        # 3. 实例化专家 (必须使用原生带 time_emb 的 ResnetBlock)
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            if expert_depth == 1:
                self.experts.append(ResnetBlock(dim, dim_out, time_emb_dim=time_emb_dim, groups=groups))
            else:
                layers = []
                for d in range(expert_depth):
                    in_d = dim if d == 0 else dim_out
                    layers.append(ResnetBlock(in_d, dim_out, time_emb_dim=time_emb_dim, groups=groups))
                self.experts.append(nn.ModuleList(layers))

    def forward(self, x, time_emb=None, sdt_weights=None):
        # 1. 如果开启了并联共享，先计算共享特征
        if self.use_shared:
            y_shared = self.shared_block(x, time_emb)
        else:
            y_shared = 0

        # 2. 专家各自提取特异性特征
        expert_outputs = []
        for expert in self.experts:
            if self.expert_depth == 1:
                out_e = expert(x, time_emb)
            else:
                out_e = x
                for layer in expert:
                    out_e = layer(out_e, time_emb)
            expert_outputs.append(out_e)

        # 3. 拦截并净化来自司令部 (SDT) 的全局概率
        clean_sdt_probs = get_sdt_probs(sdt_weights, num_experts=self.num_experts, is_logits=False)

        # 4. 核心路由决策：司令部概率 + 局部特征 = 最终权重
        # 这里的 weights 是 [B, E]
        weights, logits, feat_logits = self.router(x, sdt_probs=clean_sdt_probs)

        # =========================================================
        # 🚀 新增：微观状态探针
        # 将当前 batch、当前深度的真实分配权重缓存下来，供外部计算真实负载均衡
        # =========================================================
        self.last_weights = weights

        # 5. Soft MoE 加权融合
        y_expert = 0
        for i, exp_out in enumerate(expert_outputs):
            w = weights[:, i].view(-1, 1, 1, 1)
            y_expert += w * exp_out

        # 6. 返回最终结果 (兼容 U-Net 原生数据流格式)
        return y_shared + y_expert


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
        input_condition=False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + channels * \
            (1 if self_condition else 0) + channels * \
            (1 if condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class UnetRes(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            share_encoder=1,
            condition=False,
            input_condition=False
    ):
        super().__init__()
        self.condition = condition
        self.input_condition = input_condition
        self.share_encoder = share_encoder
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition

        # determine dimensions
        if self.share_encoder == 1:
            input_channels = channels + channels * \
                             (1 if self_condition else 0) + \
                             channels * (1 if condition else 0)
            init_dim = default(init_dim, dim)
            self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

            dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
            in_out = list(zip(dims[:-1], dims[1:]))

            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

            # time embeddings
            time_dim = dim * 4

            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features)
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

            # layers
            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            self.ups_no_skip = nn.ModuleList([])
            num_resolutions = len(in_out)

            # 准备 MoE 积木工厂 (默认使用 expert_depth=1)
            moe_block_klass = partial(
                MoEResnetBlock,
                groups=resnet_block_groups,
                num_experts=4,
                expert_depth=1,
                init_sdt_scale=0.8,
                temperature=1.0
            )

            # ======================== 构建 Downs ========================
            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (num_resolutions - 1)
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                ]))

            # ======================== 构建 Mid ========================
            mid_dim = dims[-1]
            self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)


            # ======================== 构建 Ups ========================
            num_up_stages = len(in_out)

            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (num_up_stages - 1)

                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ]))

                # =========================================================
                # 只在最高分辨率 decoder stage 使用 MoE
                # ind == num_up_stages - 1 对应最后一个 up stage，空间分辨率最高
                # 前面的低分辨率 decoder stage 使用普通 ResnetBlock，降低计算量
                # =========================================================
                if is_last:
                    second_block = moe_block_klass(
                        dim_out,
                        dim_out,
                        time_emb_dim=time_dim
                    )
                else:
                    second_block = block_klass(
                        dim_out,
                        dim_out,
                        time_emb_dim=time_dim
                    )

                self.ups_no_skip.append(nn.ModuleList([
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    second_block,
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ]))
            # ======================== 构建 Final ========================
            out_dim = default(out_dim, channels)
            self.out_dim = out_dim

            # 🚀 将 final block 升级为专家，并恢复纯粹的单一输出头
            self.final_res_block_1 = moe_block_klass(dim, dim, time_emb_dim=time_dim)
            self.final_conv_1 = nn.Conv2d(dim, self.out_dim, 1)  # 单一头出像素

            self.final_res_block_2 = moe_block_klass(dim * 2, dim, time_emb_dim=time_dim)
            self.final_conv_2 = nn.Conv2d(dim, self.out_dim, 1)

        elif self.share_encoder == 0:
            self.unet0 = Unet(dim, init_dim=init_dim, out_dim=out_dim, dim_mults=dim_mults, channels=channels,
                              self_condition=self_condition, resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance, learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim, condition=condition,
                              input_condition=input_condition)
            self.unet1 = Unet(dim, init_dim=init_dim, out_dim=out_dim, dim_mults=dim_mults, channels=channels,
                              self_condition=self_condition, resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance, learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim, condition=condition,
                              input_condition=input_condition)
        elif self.share_encoder == -1:
            self.unet0 = Unet(dim, init_dim=init_dim, out_dim=out_dim, dim_mults=dim_mults, channels=channels,
                              self_condition=self_condition, resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance, learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim, condition=condition,
                              input_condition=input_condition)

    def forward(self, x, time, x_self_cond=None, sdt_weights=None):
        if self.share_encoder == 1:
            if self.self_condition:
                x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
                x = torch.cat((x_self_cond, x), dim=1)

            x = self.init_conv(x)
            r = x.clone()
            t = self.time_mlp(time)
            h = []

            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)
                x = block2(x, t)
                x = attn(x)
                h.append(x)
                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

            out_res = x
            for block1, block2, attn, upsample in self.ups_no_skip:
                out_res = block1(out_res, t)

                if isinstance(block2, MoEResnetBlock):
                    out_res = block2(out_res, t, sdt_weights=sdt_weights)
                else:
                    out_res = block2(out_res, t)

                out_res = attn(out_res)
                out_res = upsample(out_res)
            # 🚀 最后一层：高维专家处理特征，单头输出像素
            out_res = self.final_res_block_1(out_res, t, sdt_weights=sdt_weights)
            out_res_conv1 = self.final_conv_1(out_res)

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)
                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x)
                x = upsample(x)

            x = torch.cat((x, r), dim=1)
            # 🚀 第二个分支同理
            x = self.final_res_block_2(x, t, sdt_weights=sdt_weights)
            out_res_add_noise = self.final_conv_2(x)

            return out_res_conv1, out_res_add_noise

        elif self.share_encoder == 0:
            return self.unet0(x, time, x_self_cond=x_self_cond), self.unet1(x, time, x_self_cond=x_self_cond)
        elif self.share_encoder == -1:
            return [self.unet0(x, time, x_self_cond=x_self_cond)]

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        x = torch.flip(x, dims=[0])
        scale = 0.5*timesteps*(timesteps+1)
        alphas = x/scale
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float64)
    assert alphas.sum()-torch.tensor(1) < torch.tensor(1e-10)

    return alphas*sum_scale


class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        input_condition=False,
        input_condition_mask=False
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask
        self.last_print_step = -1
        self.last_expert_prior_loss = None
        self.last_expert_prior_weight = None
        # 【新增代码】：当开启 input_condition 时，初始化软决策树
        if self.input_condition:
            self.sdt = SoftDecisionTree(in_channels=self.channels, num_experts=4)

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        alphas = gen_coefficients(timesteps, schedule="decreased")
        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2 = gen_coefficients(
            timesteps, schedule="increased", sum_scale=self.sum_scale)
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
        betas_cumsum = torch.sqrt(betas2_cumsum)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1 - alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1', betas2_cumsum_prev / betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 * alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2 / betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6




    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t-x_input-(extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, sdt_weights=None, x_self_cond=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  t,
                                  x_self_cond,
                                  sdt_weights=sdt_weights)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(
                x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_res_add_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(
                x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == 'pred_x0_add_noise':
            x_start = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = x_input-x_start
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, sdt_weights=None, x_self_cond=None):
        preds = self.model_predictions(
            x_input, x, t, sdt_weights=sdt_weights, x_self_cond=x_self_cond)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, sdt_weights=None, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, sdt_weights=sdt_weights, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        if self.input_condition:
            # 🚀 兼容三输入采样提取
            sdt_in = x_input[1] if isinstance(x_input, list) and len(x_input) > 1 else x_input[0]
            # 【强制满进度】：测试时保证最低温
            sdt_weights = self.sdt(sdt_in, current_step=1, total_steps=1)
        else:
            sdt_weights = None
        x_input = x_input[0] if isinstance(x_input, list) else x_input

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, sdt_weights=sdt_weights, x_self_cond=self_cond)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps=False, imgs=None):
        if self.input_condition and imgs is not None:
            # 从传入的 imgs 列表中提取低质图像 x_input
            x_input_img = imgs[0] if isinstance(imgs, list) else imgs

            # 🚀 兼容三输入采样提取
            sdt_in = imgs[1] if isinstance(imgs, list) and len(imgs) > 1 else x_input_img

            # 【强制满进度】：告诉 SDT 现在是最后一步，强行切入最低温的开发期！
            sdt_weights = self.sdt(sdt_in, current_step=1, total_steps=1)
            x_input = x_input_img
        else:
            x_input = 0
            sdt_weights = None

        # ==========================================================
        # 2. 采样基础参数初始化
        # ==========================================================
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        # 补齐丢失的 last 变量定义
        last = not return_all_timesteps
        if not last:
            img_list = []

        # ==========================================================
        # 3. DDIM 加速跳步采样循环
        # ==========================================================
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            # 【核心修改】：精准地将 sdt_weights 传给模型
            preds = self.model_predictions(
                x_input=x_input,
                x=img,
                t=time_cond,
                sdt_weights=sdt_weights,  # <--- SDT 权重在这里发力！
                x_self_cond=self_cond
            )

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum - alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                                                                                betas2_cumsum_next - sigma2).sqrt() / betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            if type == "use_pred_noise":
                img = img - alpha * pred_res - \
                      (betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()) * \
                      pred_noise + sigma2.sqrt() * noise
            elif type == "use_x_start":
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum * img + \
                      (1 - sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * x_start + \
                      (
                                  alpha_cumsum_next - alpha_cumsum * sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * pred_res + \
                      sigma2.sqrt() * noise
            elif type == "special_eta_0":
                img = img - alpha * pred_res - \
                      (betas_cumsum - betas_cumsum_next) * pred_noise
            elif type == "special_eta_1":
                img = img - alpha * pred_res - betas2 / betas_cumsum * pred_noise + \
                      betas * betas2_cumsum_next.sqrt() / betas_cumsum * noise

            if not last:
                img_list.append(img)

        # ==========================================================
        # 4. 组装输出
        # ==========================================================
        if self.condition:
            if not last:
                img_list = [input_add_noise] + img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)
    







    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):
        image_size, channels = self.image_size, self.channels

        if self.condition:
            if self.input_condition and self.input_condition_mask:
                x_input[0] = normalize_to_neg_one_to_one(x_input[0])
            else:
                x_input = normalize_to_neg_one_to_one(x_input)
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)

        # 【核心修改：明确分发，不再强行统一传参】
        if not self.is_ddim_sampling:
            # 原生全步长采样
            return self.p_sample_loop(x_input=x_input, shape=size, last=last)
        else:
            # DDIM 加速采样
            return self.ddim_sample(shape=size, return_all_timesteps=not last, imgs=x_input)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise
        )




    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
        

    def get_expert_prior_weight(self, current_step=None):
        """
        expert target loss 衰减：
        0~2k:      0.01
        2k~10k:    0.01 -> 0.002
        10k以后:   0.001
        """
        if current_step is None:
            step = int(getattr(self, "current_train_step", 0))
        else:
            step = int(current_step)

        if step < 2000:
            return 0.01

        if step < 10000:
            ratio = (step - 2000) / float(10000 - 2000)
            return 0.01 * (1.0 - ratio) + 0.002 * ratio

        return 0.001


    def get_balance_weight(self, current_step=None):
        """
        load balance 只作为防塌缩约束，不要太强。
        """
        if current_step is None:
            step = int(getattr(self, "current_train_step", 0))
        else:
            step = int(current_step)

        if step < 5000:
            return 0.003

        return 0.001


    def kl_prob_loss(self, probs, target):
        """
        probs:  [B, E], 已经 softmax
        target: [B, E], soft target
        """
        probs = probs.clamp_min(1e-8)
        target = target.clamp_min(1e-8)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return F.kl_div(
            probs.log(),
            target,
            reduction="batchmean"
        )


    def collect_moe_weights(self):
        """
        收集所有 MoEResnetBlock 的 last_weights。
        每个 MoEResnetBlock forward 后会缓存 last_weights。
        """
        weights = []
        for m in self.model.modules():
            if isinstance(m, MoEResnetBlock) and hasattr(m, "last_weights"):
                weights.append(m.last_weights)

        return weights


    def compute_expert_prior_loss(self, expert_target, sdt_weights=None):
        """
        同时监督：
        1) SDT 全局路由权重
        2) MoE block 真实专家权重

        这样能让 SDT / MoE 前期学会按 sigma 分工。
        """
        if expert_target is None:
            return None, None

        expert_target = expert_target.float()
        expert_target = expert_target / expert_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        loss_list = []

        # 1) 监督 SDT 权重
        if sdt_weights is not None:
            sdt_probs = get_sdt_probs(sdt_weights, num_experts=expert_target.shape[-1], is_logits=False)
            if sdt_probs is not None:
                loss_list.append(self.kl_prob_loss(sdt_probs, expert_target))

        # 2) 监督各个 MoE block 的真实权重
        moe_weights = self.collect_moe_weights()
        for w in moe_weights:
            loss_list.append(self.kl_prob_loss(w, expert_target))

        if len(loss_list) == 0:
            return None, None

        expert_prior_loss = sum(loss_list) / len(loss_list)

        # 3) 弱 load balance，防止塌缩
        balance_losses = []
        for w in moe_weights:
            mean_usage = w.mean(dim=0)
            uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
            balance_losses.append(F.mse_loss(mean_usage, uniform))

        if len(balance_losses) > 0:
            balance_loss = sum(balance_losses) / len(balance_losses)
        else:
            balance_loss = None

        return expert_prior_loss, balance_loss


    def p_losses(self, imgs, t, noise=None, current_step=0, total_steps=None):
        total_steps = 20000 if total_steps is None else total_steps

        # 记录当前训练步数，供 expert prior / balance loss 衰减使用
        self.current_train_step = int(current_step)
        self.total_train_steps = int(total_steps)

        if isinstance(imgs, list):  # Condition
            x_start = imgs[0]
            x_input = imgs[1]

            x_extra_cond = None
            expert_target = None

            if len(imgs) >= 3:
                # [B,C,H,W] 视作额外图像条件；[B,4] 视作 expert_target
                if torch.is_tensor(imgs[2]) and imgs[2].dim() == 4:
                    x_extra_cond = imgs[2]
                elif torch.is_tensor(imgs[2]) and imgs[2].dim() == 2:
                    expert_target = imgs[2]

            if len(imgs) >= 4:
                if torch.is_tensor(imgs[3]) and imgs[3].dim() == 2:
                    expert_target = imgs[3]

            if self.input_condition:
                sdt_in = x_extra_cond if x_extra_cond is not None else x_input
                sdt_weights = self.sdt(sdt_in, current_step=current_step, total_steps=total_steps)
            else:
                sdt_weights = None
        else:  # Generation
            x_input = 0
            x_start = imgs
            x_extra_cond = None
            expert_target = None
            sdt_weights = None


        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        # noise sample
        x = self.q_sample(x_start, x_res, t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x_input, x, t, sdt_weights=sdt_weights if self.input_condition else None).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)


        model_out = self.model(x_in,
                               t,
                               x_self_cond,
                               sdt_weights=sdt_weights)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_res_add_noise':
            target.append(x_res)
            target.append(x_res+noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]-model_out[0]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_add_noise':
            target.append(x_start)
            target.append(x_start+noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1] - model_out[0]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)

            pred_res = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
        else:
            loss = 0
            for i in range(len(model_out)):
                loss = loss + \
                    self.loss_fn(model_out[i], target[i], reduction='none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        total_loss = loss.mean()
        # ==========================================================
        # Sigma soft expert target + decayed expert prior loss
        # 作用：
        # 前期用 sigma soft target 引导 SDT/MoE 专家分工；
        # 后期逐步减弱，让路由更多依靠图像特征。
        # ==========================================================
        if self.input_condition and expert_target is not None:
            expert_target = expert_target.to(total_loss.device)

            expert_prior_loss, balance_loss = self.compute_expert_prior_loss(
                expert_target=expert_target,
                sdt_weights=sdt_weights
            )

            # 1) expert prior loss：按 step 衰减
            if expert_prior_loss is not None:
                lambda_expert_prior = self.get_expert_prior_weight(current_step)
                total_loss = total_loss + lambda_expert_prior * expert_prior_loss

                self.last_expert_prior_loss = float(expert_prior_loss.detach().item())
                self.last_expert_prior_weight = float(lambda_expert_prior)
            else:
                self.last_expert_prior_loss = None
                self.last_expert_prior_weight = None

            # 2) weak load balance：只防止塌缩，不强行均匀
            if balance_loss is not None:
                lambda_balance = self.get_balance_weight(current_step)
                total_loss = total_loss + lambda_balance * balance_loss
            else:
                lambda_balance = None

            # 3) 打印检查
            if current_step % 1000 == 0 and current_step != self.last_print_step:
                moe_weights = self.collect_moe_weights()
                if len(moe_weights) > 0:
                    all_weights = torch.cat(moe_weights, dim=0)
                    mean_probs = all_weights.mean(dim=0).detach().cpu().numpy()
                else:
                    mean_probs = None

                print(
                    f"[MoE监督] step={current_step} "
                    f"mean_probs={mean_probs} "
                    f"expert_prior={self.last_expert_prior_loss} "
                    f"lambda_prior={self.last_expert_prior_weight} "
                    f"balance={float(balance_loss.detach().item()) if balance_loss is not None else None} "
                    f"lambda_balance={lambda_balance}"
                )
                self.last_print_step = current_step

        else:
            self.last_expert_prior_loss = None
            self.last_expert_prior_weight = None

        return total_loss



    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size = *img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size = *img.shape, img.device, self.image_size

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if isinstance(img, list):
            normed = []
            for item in img:
                # 只归一化图像张量 [B,C,H,W]，不动 expert_target [B,4]
                if torch.is_tensor(item) and item.dim() == 4:
                    normed.append(item * 2.0 - 1.0)
                else:
                    normed.append(item)
            img = normed
        else:
            img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)

# trainer class


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results/sample',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
        equalizeHist=False,
        crop_patch=False,
        generation=False
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.crop_patch = crop_patch

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition

        if self.condition:
            if len(folder) == 3:
                self.condition_type = 1
                # test_input
                ds = dataset(folder[-1], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                               pin_memory=True, num_workers=0)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=0)))
            elif len(folder) == 4:
                self.condition_type = 2
                # test_gt+test_input
                ds = dataset(folder[2:4], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=False, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds
                self.val_loader = DataLoader(
                    self.sample_dataset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=0
                )
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=False,
                                                                               pin_memory=True, num_workers=0)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=0)))
            elif len(folder) == 6:
                self.condition_type = 3
                # test_gt+test_input
                ds = dataset(folder[3:6], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:3]

                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                               pin_memory=True, num_workers=0)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=0)))
        else:
            self.condition_type = 0
            trian_folder = folder

            ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                         convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
            self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                            shuffle=True, pin_memory=True, num_workers=0)))

        # optimizer

        self.opt = Adam(diffusion_model.parameters(),
                        lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

            # =======================================================
            # 🚀 新增：初始化 TensorBoard 记录器
            # =======================================================
            self.writer = SummaryWriter(log_dir=str(self.results_folder / 'logs'))
        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        device = self.accelerator.device
        self.device = device


    def format_seconds(self, seconds):
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"
    def get_eval_dataset_name(self):
        """
        从 sample_dataset 的 GT 路径里推断当前验证/测试集名字。
        例如:
            .../benchmark/DF2KVal/HR/xxx.png -> DF2KVal
            .../benchmark/Set5/HR/xxx.png    -> Set5
        """
        if not hasattr(self, 'sample_dataset'):
            return "Unknown"

        if not hasattr(self.sample_dataset, 'gt'):
            return "Unknown"

        if len(self.sample_dataset.gt) == 0:
            return "Unknown"

        p = str(self.sample_dataset.gt[0]).replace("\\", "/")
        parts = p.split("/")

        if "benchmark" in parts:
            idx = parts.index("benchmark")
            if idx + 1 < len(parts):
                return parts[idx + 1]

        return "Unknown"

    def is_public_benchmark(self, dataset_name):
        public_sets = {
            "DF2KVal",
            "Set5",
            "Set14",
            "B100",
            "BSDS100",
            "Urban100",
            "manga109",
            "Manga109"
        }
        return dataset_name in public_sets

    def quantize_255(self, img):
        """
        输入 img 为 [0,1] tensor
        输出为 [0,255] tensor, round + clamp
        对齐 MCDFormer utility.quantize(..., rgb_range=255)
        """
        return img.mul(255.0).clamp(0, 255).round()

    def remove_pad(self, img, pad_size):
        """
        pad_size = [bottom, right]
        img: [B,C,H,W]
        """
        bottom, right = pad_size
        _, _, h, w = img.shape

        h_end = h - bottom if bottom > 0 else h
        w_end = w - right if right > 0 else w

        return img[:, :, :h_end, :w_end]

    def crop_pair_to_scale(self, sr, hr, scale):
        """
        对齐 MCDFormer 的 crop_border 思路：
        保证尺寸能整除 scale
        """
        _, _, h, w = hr.shape
        h = int(h // scale * scale)
        w = int(w // scale * scale)

        sr = sr[:, :, :h, :w]
        hr = hr[:, :, :h, :w]
        return sr, hr

    def calc_psnr_mcd(self, sr_255, hr_255, scale, benchmark=False):
        """
        复刻 MCDFormer utility.calc_psnr 逻辑
        输入:
            sr_255, hr_255: [B,C,H,W], range [0,255]
        """
        diff = (sr_255 - hr_255).div(255.0)

        if benchmark:
            shave = scale
            if diff.size(1) > 1:
                convert = diff.new_zeros(1, 3, 1, 1)
                convert[0, 0, 0, 0] = 65.738
                convert[0, 1, 0, 0] = 129.057
                convert[0, 2, 0, 0] = 25.064
                diff = diff.mul(convert).div(256.0)
                diff = diff.sum(dim=1, keepdim=True)
        else:
            shave = scale + 6

        shave = int(math.ceil(shave))

        if diff.size(-2) <= 2 * shave or diff.size(-1) <= 2 * shave:
            valid = diff
        else:
            valid = diff[:, :, shave:-shave, shave:-shave]

        mse = valid.pow(2).mean().item()
        if mse <= 1e-12:
            return 100.0

        return -10 * math.log10(mse)

    def calc_ssim_mcd(self, sr_255, hr_255, scale=2, benchmark=False):
        """
        复刻 MCDFormer utility.calc_ssim 的主逻辑：
        - 输入为 [0,255]
        - 转 Y 通道
        - 再做 SSIM
        """
        border = int(math.ceil(scale)) if benchmark else int(math.ceil(scale) + 6)

        img1 = sr_255.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = hr_255.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
        img2 = np.transpose(img2, (1, 2, 0))

        img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
        img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0

        if img1_y.shape != img2_y.shape:
            raise ValueError("Input images must have the same dimensions.")

        h, w = img1_y.shape[:2]
        if h <= 2 * border or w <= 2 * border:
            crop1 = img1_y
            crop2 = img2_y
        else:
            crop1 = img1_y[border:h - border, border:w - border]
            crop2 = img2_y[border:h - border, border:w - border]

        return self.ssim_np(crop1, crop2)

    def ssim_np(self, img1, img2):
        """
        与 MCDFormer utility.ssim 一致
        输入为单通道 2D numpy array, range [0,255]
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(ssim_map.mean())
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')

        if path.exists():
            data = torch.load(
                str(path), map_location=self.device)

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            #self.ema.load_state_dict(data['ema'])
            ema_state = data['ema']
            if 'initted' in ema_state and ema_state['initted'].shape == torch.Size([]):
                ema_state['initted'] = ema_state['initted'].unsqueeze(0)
            if 'step' in ema_state and ema_state['step'].shape == torch.Size([]):
                ema_state['step'] = ema_state['step'].unsqueeze(0)
            self.ema.load_state_dict(ema_state)

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - "+str(path))

        self.ema.to(self.device)

    def train(self):
        accelerator = self.accelerator

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            avg_step_time = None
            time_ema = 0.9

            while self.step < self.train_num_steps:

                step_start_time = time.time()
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data, current_step=self.step, total_steps=self.train_num_steps)
                        loss = loss / self.gradient_accumulate_every
                        total_loss = total_loss + loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1



                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    # =======================================================
                    # 🚀 新增：实时记录 Training Loss 及 MoE 专家状态
                    # =======================================================
                    self.writer.add_scalar('train/loss', total_loss, self.step)

                    # 1. 获取并解包真实模型，防止 DDP 或 EMA 外壳阻挡属性访问
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    if hasattr(unwrapped_model, 'model'):  # 剥离 ResidualDiffusion 外壳，触达内部 UnetRes
                        core_unet = unwrapped_model.model
                    else:
                        core_unet = unwrapped_model
                    if hasattr(unwrapped_model,
                               'last_expert_prior_loss') and unwrapped_model.last_expert_prior_loss is not None:
                        self.writer.add_scalar("train/loss_expert_prior", unwrapped_model.last_expert_prior_loss,
                                               self.step)

                    if hasattr(unwrapped_model,
                               'last_expert_prior_weight') and unwrapped_model.last_expert_prior_weight is not None:
                        self.writer.add_scalar("train/lambda_expert_prior", unwrapped_model.last_expert_prior_weight,
                                               self.step)
                    # 2. 扫描底层网络，精准拦截真实微观权重 (兼容断点续训 step)
                    if self.step % 100 == 0:
                        real_moe_weights = []
                        for module in core_unet.modules():
                            if type(module).__name__ == 'MoEResnetBlock':
                                if hasattr(module, 'last_weights') and module.last_weights is not None:
                                    real_moe_weights.append(module.last_weights.detach())

                        if len(real_moe_weights) > 0:
                            # [N_layers * B, E]
                            all_weights = torch.cat(real_moe_weights, dim=0)

                            # 1. 平均负载：w1~w4
                            mean_weights = all_weights.mean(dim=0)
                            actual_num_experts = mean_weights.shape[0]
                            for i in range(actual_num_experts):
                                self.writer.add_scalar(f"train/w{i + 1}", mean_weights[i].item(), self.step)

                            # 2. 最大专家权重
                            max_weight = all_weights.max(dim=-1).values.mean()
                            self.writer.add_scalar("train/max_weight", max_weight.item(), self.step)

                            # 3. 熵
                            safe_weights = all_weights.clamp_min(1e-8)
                            entropy = -(safe_weights * safe_weights.log()).sum(dim=-1).mean()
                            self.writer.add_scalar("train/entropy", entropy.item(), self.step)

                        # 4. SDT 实际退火温度
                        sdt_temp = None
                        if hasattr(unwrapped_model, "sdt") and hasattr(unwrapped_model.sdt, "last_temperature"):
                            sdt_temp = unwrapped_model.sdt.last_temperature

                        if sdt_temp is not None:
                            self.writer.add_scalar("train/sdt_temp", float(sdt_temp), self.step)

                        # 5. SDT 当前进度（可选）
                        if hasattr(unwrapped_model, "sdt") and hasattr(unwrapped_model.sdt, "last_progress"):
                            sdt_progress = unwrapped_model.sdt.last_progress
                            if sdt_progress is not None:
                                self.writer.add_scalar("train/sdt_progress", float(sdt_progress), self.step)



                    if self.step % 200 == 0:
                        self.writer.flush()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.sample(milestone)
                        self.validate_fixed(max_val_images=20)
                        self.save(milestone)


                step_time = time.time() - step_start_time

                if avg_step_time is None:
                    avg_step_time = step_time
                else:
                    avg_step_time = time_ema * avg_step_time + (1 - time_ema) * step_time

                remaining_steps = self.train_num_steps - self.step
                eta_seconds = avg_step_time * remaining_steps

                pbar.set_description(
                    f'loss: {total_loss:.4f} | step_time: {step_time:.2f}s | eta: {self.format_seconds(eta_seconds)}'
                )
                pbar.update(1)

        accelerator.print('training complete')


    def get_eval_dataset_name(self):
        if not hasattr(self, "sample_dataset"):
            return "Unknown"
        if not hasattr(self.sample_dataset, "gt"):
            return "Unknown"
        if len(self.sample_dataset.gt) == 0:
            return "Unknown"

        p = str(self.sample_dataset.gt[0]).replace("\\", "/")
        parts = p.split("/")

        if "benchmark" in parts:
            idx = parts.index("benchmark")
            if idx + 1 < len(parts):
                return parts[idx + 1]

        return "Unknown"



    def quantize_255(self, img):
        """
        输入 [0,1]，输出 [0,255]，round + clamp。
        对齐 MCDFormer utility.quantize 的效果。
        """
        return img.mul(255.0).clamp(0, 255).round()


    def remove_pad(self, img, pad_size):
        bottom, right = pad_size
        _, _, h, w = img.shape

        h_end = h - bottom if bottom > 0 else h
        w_end = w - right if right > 0 else w

        return img[:, :, :h_end, :w_end]


    def crop_pair_to_scale(self, sr, hr, scale=4):
        """
        保证 SR / HR 尺寸能被 scale 整除。
        """
        _, _, h, w = hr.shape
        h = int(h // scale * scale)
        w = int(w // scale * scale)

        sr = sr[:, :, :h, :w]
        hr = hr[:, :, :h, :w]
        return sr, hr


    def calc_psnr_mcd(self, sr_255, hr_255, scale=4, benchmark=True):
        """
        对齐 MCDFormer 的 calc_psnr:
        - benchmark=True: shave = scale，且转 Y 通道
        - benchmark=False: shave = scale + 6
        """
        diff = (sr_255 - hr_255).div(255.0)

        if benchmark:
            shave = scale
            if diff.size(1) > 1:
                convert = diff.new_zeros(1, 3, 1, 1)
                convert[0, 0, 0, 0] = 65.738
                convert[0, 1, 0, 0] = 129.057
                convert[0, 2, 0, 0] = 25.064
                diff = diff.mul(convert).div(256.0)
                diff = diff.sum(dim=1, keepdim=True)
        else:
            shave = scale + 6

        shave = int(math.ceil(shave))

        if diff.size(-2) <= 2 * shave or diff.size(-1) <= 2 * shave:
            valid = diff
        else:
            valid = diff[:, :, shave:-shave, shave:-shave]

        mse = valid.pow(2).mean().item()
        if mse <= 1e-12:
            return 100.0

        return -10 * math.log10(mse)


    def calc_ssim_mcd(self, sr_255, hr_255, scale=4, benchmark=True):
        """
        对齐 MCDFormer 的 SSIM:
        - 输入 [0,255]
        - 转 Y 通道
        - benchmark=True 时 border=scale
        """
        border = int(math.ceil(scale)) if benchmark else int(math.ceil(scale) + 6)

        img1 = sr_255.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
        img2 = hr_255.data.squeeze().float().clamp(0, 255).round().cpu().numpy()

        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

        img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
        img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0

        h, w = img1_y.shape[:2]
        if h > 2 * border and w > 2 * border:
            img1_y = img1_y[border:h - border, border:w - border]
            img2_y = img2_y[border:h - border, border:w - border]

        return self.ssim_np(img1_y, img2_y)


    def ssim_np(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(ssim_map.mean())




    @torch.no_grad()
    def validate_fixed(self, max_val_images=20):
        """
        按 MCDFormer benchmark 风格统计验证集指标：
        - pred / gt 转 [0,255]
        - PSNR / SSIM 同一口径
        - DF2KVal 走 benchmark=False
        - Set5 / Set14 / BSDS100 / Urban100 走 benchmark=True
        """
        self.ema.ema_model.eval()

        dataset_name = self.get_eval_dataset_name()
        benchmark_mode = self.is_public_benchmark(dataset_name)

        psnr_list = []
        ssim_list = []

        for i, items in enumerate(self.val_loader):
            if i >= max_val_images:
                break

            if self.condition_type in [2, 3]:
                batch = [item.to(self.device) for item in items]
                gt = batch[0]
                x_input_sample = batch[1:]
            else:
                continue

            outputs = list(
                self.ema.ema_model.sample(
                    x_input_sample,
                    batch_size=1,
                    last=True
                )
            )
            pred = outputs[-1]

            # 去掉 pad
            pad_size = self.sample_dataset.get_pad_size(i)
            gt = self.remove_pad(gt, pad_size)
            pred = self.remove_pad(pred, pad_size)

            # 保证尺寸能整除 scale
            scale = 4

            pred, gt = self.crop_pair_to_scale(pred, gt, scale)

            # 转到 [0,255]
            pred_255 = self.quantize_255(pred)
            gt_255 = self.quantize_255(gt)

            psnr = self.calc_psnr_mcd(pred_255, gt_255, scale=scale, benchmark=benchmark_mode)
            ssim = self.calc_ssim_mcd(pred_255, gt_255, scale=scale, benchmark=benchmark_mode)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

        if len(psnr_list) > 0:
            mean_psnr = sum(psnr_list) / len(psnr_list)
            mean_ssim = sum(ssim_list) / len(ssim_list)

            self.writer.add_scalar("Metrics_Validation/PSNR_fixed", mean_psnr, self.step)
            self.writer.add_scalar("Metrics_Validation/SSIM_fixed", mean_ssim, self.step)

            print(
                f"[Validation:{dataset_name}] step={self.step} "
                f"PSNR={mean_psnr:.4f} dB, SSIM={mean_ssim:.4f} "
                f"over {len(psnr_list)} images (benchmark_mode={benchmark_mode})"
            )
    def sample(self, milestone, last=True, FID=False):
        self.ema.ema_model.eval()

        with torch.no_grad():
            batches = self.num_samples
            # 提取输入逻辑保持不变
            if self.condition_type == 0:
                x_input_sample = [0]
                show_x_input_sample = []
            elif self.condition_type == 1:
                x_input_sample = [next(self.sample_loader).to(self.device)]
                show_x_input_sample = x_input_sample
            elif self.condition_type == 2 or self.condition_type == 3:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device) for item in x_input_sample]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]

            # --- 优化点：只运行一次采样 ---
            sampled_outputs = list(self.ema.ema_model.sample(
                x_input_sample, batch_size=batches, last=last))

            final_output = sampled_outputs[-1]  # 模型修复结果
            
            if self.condition_type == 2:
                corrupted_input = show_x_input_sample[1]
            elif self.condition_type == 3:
                corrupted_input = show_x_input_sample[1]
            elif len(show_x_input_sample) > 0:
                corrupted_input = show_x_input_sample[-1]
            else:
                corrupted_input = final_output
            
        # =======================================================
        # Validation metrics + visualization
        # =======================================================
        if self.condition_type > 0:
            gt_img = show_x_input_sample[0]
        else:
            gt_img = None

        if self.condition_type in [2, 3]:
            corrupted_input = show_x_input_sample[1]
        elif len(show_x_input_sample) > 0:
            corrupted_input = show_x_input_sample[-1]
        else:
            corrupted_input = final_output

        # 1) 记录“当前 sample 批次”的 PSNR
        if gt_img is not None:
            mse = F.mse_loss(final_output, gt_img)
            psnr = -10 * torch.log10(mse + 1e-8) if mse > 0 else torch.tensor(100.0, device=gt_img.device)

            # 建议把这个名字改掉，避免和固定验证集 PSNR 混淆
            self.writer.add_scalar('Metrics_Validation/PSNR_sample', psnr.item(), self.step)

        # 2) 恢复误差图：Output vs GT
        if gt_img is not None:
            error_map = torch.abs(final_output - gt_img) * 10
            error_map = torch.clamp(error_map, 0, 1)

            # 拼图顺序：GT / Input / Output / ErrorMap
            all_images_list = [gt_img, corrupted_input, final_output, error_map]
        else:
            # 无 GT 时退回到输入差异图
            diff_map = torch.abs(final_output - corrupted_input) * 10
            diff_map = torch.clamp(diff_map, 0, 1)
            all_images_list = [corrupted_input, final_output, diff_map]
            all_images = torch.cat(all_images_list, dim=0)

            # 自动调整 nrow，让每一组对比都在一行显示
            # 如果你有 GT 和 Input，再加上 Output 和 Residual，一行应该显示 4 张
            num_per_group = len(all_images_list)
            nrow = num_per_group if not last else int(math.sqrt(self.num_samples)) * num_per_group

            file_name = f'sample-{milestone}.png'
            utils.save_image(all_images, str(self.results_folder / file_name), nrow=nrow)
            print("sampe-save " + file_name)
        return milestone

    def test(self, sample=False, last=True, FID=False):
        """
        按 MCDFormer benchmark 风格测试：
        - 逐张保存 SR 图
        - 逐张打印 PSNR / SSIM
        - 最后汇总平均 PSNR / SSIM
        """
        print("test start")

        if self.condition:
            self.ema.ema_model.eval()

            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=0
            )

            dataset_name = self.get_eval_dataset_name()
            benchmark_mode = self.is_public_benchmark(dataset_name)

            psnr_all = []
            ssim_all = []

            for i, items in enumerate(tqdm(loader, desc=f"testing {dataset_name}", total=len(self.sample_dataset))):
                if self.condition:
                    file_name = self.sample_dataset.load_name(i, sub_dir=self.sub_dir)
                else:
                    file_name = f"{i}.png"

                with torch.no_grad():
                    if self.condition_type == 0:
                        x_input_sample = [0]
                        show_x_input_sample = []
                    elif self.condition_type == 1:
                        x_input_sample = [items.to(self.device)]
                        show_x_input_sample = x_input_sample
                    elif self.condition_type in [2, 3]:
                        x_input_sample = [item.to(self.device) for item in items]
                        show_x_input_sample = x_input_sample
                        x_input_sample = x_input_sample[1:]

                    outputs = list(
                        self.ema.ema_model.sample(
                            x_input_sample,
                            batch_size=1,
                            last=last
                        )
                    )
                    pred = outputs[-1]

                    gt = None
                    lq_input = None
                    if self.condition_type in [2, 3]:
                        gt = show_x_input_sample[0]
                        lq_input = show_x_input_sample[1]

                # 去 pad
                pad_size = self.sample_dataset.get_pad_size(i)
                pred = self.remove_pad(pred, pad_size)

                if gt is not None:
                    gt = self.remove_pad(gt, pad_size)
                if lq_input is not None:
                    lq_input = self.remove_pad(lq_input, pad_size)

                # crop 到可整除 scale
                scale = 4
                if gt is not None:
                    pred, gt = self.crop_pair_to_scale(pred, gt, scale)
                else:
                    _, _, h, w = pred.shape
                    h = int(h // scale * scale)
                    w = int(w // scale * scale)
                    pred = pred[:, :, :h, :w]

                # 保存结果图（保存 0~1 的 pred 即可）
                save_path = str(self.results_folder / file_name)
                utils.save_image(pred, save_path, nrow=1)

                # 计算指标
                if gt is not None:
                    pred_255 = self.quantize_255(pred)
                    gt_255 = self.quantize_255(gt)

                    psnr = self.calc_psnr_mcd(
                        pred_255, gt_255,
                        scale=scale,
                        benchmark=benchmark_mode
                    )
                    ssim = self.calc_ssim_mcd(
                        pred_255, gt_255,
                        scale=scale,
                        benchmark=benchmark_mode
                    )

                    psnr_all.append(psnr)
                    ssim_all.append(ssim)

                    print(
                        f"[Test:{dataset_name}] [{i + 1:04d}/{len(self.sample_dataset):04d}] "
                        f"{file_name} | PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f}"
                    )
                else:
                    print(
                        f"[Test:{dataset_name}] [{i + 1:04d}/{len(self.sample_dataset):04d}] "
                        f"{file_name} | saved"
                    )

            if len(psnr_all) > 0:
                mean_psnr = sum(psnr_all) / len(psnr_all)
                mean_ssim = sum(ssim_all) / len(ssim_all)

                print("\n================ Final Benchmark Result ================")
                print(
                    f"{dataset_name} | Mean PSNR: {mean_psnr:.4f} dB | "
                    f"Mean SSIM: {mean_ssim:.4f} | benchmark_mode={benchmark_mode}"
                )

                metric_file = self.results_folder / f"metrics_{dataset_name}.txt"
                with open(metric_file, "w", encoding="utf-8") as f:
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Benchmark mode: {benchmark_mode}\n")
                    f.write(f"Mean PSNR: {mean_psnr:.6f} dB\n")
                    f.write(f"Mean SSIM: {mean_ssim:.6f}\n")
                    f.write(f"Images: {len(psnr_all)}\n")

                print(f"Metrics saved to: {metric_file}")

        else:
            if FID:
                self.total_n_samples = 50000
                img_id = len(glob.glob(f"{self.results_folder}/*"))
                n_rounds = (self.total_n_samples - img_id) // self.num_samples + 1
            else:
                n_rounds = 100

            for i in range(n_rounds):
                if FID:
                    i = img_id
                img_id = self.sample(i, last=last, FID=FID)

        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        # 使用 pathlib 递归创建文件夹，安全且彻底解决 FileNotFoundError
        self.results_folder.mkdir(parents=True, exist_ok=True)
