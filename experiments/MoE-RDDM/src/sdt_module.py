import torch
import torch.nn as nn
import math # <--- 新增：用于余弦退火计算
import random
import torch.nn.functional as F

class SoftDecisionTree(nn.Module):
    def __init__(self, in_channels=3, num_experts=4):
        super(SoftDecisionTree, self).__init__()

        # --- 多尺度截断特征提取 ---
        self.stage1_2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.stage3_4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # --- SDT 内部决策节点网络 ---
        self.tree_nodes = nn.Sequential(
            nn.Linear(320, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 3)  # 输出 3 个节点的 Logit
        )

        # --- 🌟 温度退火超参数设定 ---
        self.temp_max = 1.0    # 初始最高温度
        self.temp_min = 0.2    # 最终最低温度
        self.warmup_ratio = 0.1 # 前 10% 步数为高温探索期
        self.hold_ratio = 0.9   # 前 90% 完成退火，最后 10% 保持低温

        # --- 运行时监控缓存 ---
        self.last_temperature = None
        self.last_progress = None
        self.last_stage = None

    def forward(self, x, current_step=0, total_steps=20000):
        # total_steps 应与 train.py 中的 train_num_steps 保持一致

        # 1. 特征提取与融合
        f1 = self.stage1_2(x)
        f2 = self.stage3_4(f1)

        avg1 = self.avg_pool(f1).flatten(1)
        max1 = self.max_pool(f1).flatten(1)
        p1 = torch.cat([avg1, max1], dim=1)

        avg2 = self.avg_pool(f2).flatten(1)
        max2 = self.max_pool(f2).flatten(1)
        p2 = torch.cat([avg2, max2], dim=1)

        combined = torch.cat([p1, p2], dim=1)  # [Batch, 320]

        # --- 🌟 核心升级：三阶段余弦温度退火 ---
        progress = min(current_step / max(total_steps, 1), 1.0)

        if progress < self.warmup_ratio:
            # 阶段 1：高温探索期
            temp = self.temp_max
            stage_name = "高温"
        elif progress > self.hold_ratio:
            # 阶段 3：低温开发期
            temp = self.temp_min
            stage_name = "低温"
        else:
            # 阶段 2：余弦退火期
            # 计算在退火阶段内的相对进度 (0.0 到 1.0)
            decay_progress = (progress - self.warmup_ratio) / (self.hold_ratio - self.warmup_ratio)
            # 余弦衰减公式: 0.5 * (max - min) * (1 + cos(pi * progress)) + min
            temp = self.temp_min + 0.5 * (self.temp_max - self.temp_min) * (1 + math.cos(math.pi * decay_progress))
            stage_name = "余弦退火"

        # 2. 树状概率流计算
        node_logits = self.tree_nodes(combined)
        node_probs = torch.sigmoid(node_logits / temp)

        p_root = node_probs[:, 0:1]
        p_left = node_probs[:, 1:2]
        p_right = node_probs[:, 2:3]

        w1 = p_root * p_left
        w2 = p_root * (1.0 - p_left)
        w3 = (1.0 - p_root) * p_right
        w4 = (1.0 - p_root) * (1.0 - p_right)

        weights = torch.cat([w1, w2, w3, w4], dim=1)  # [Batch, 4]

        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        self.last_temperature = float(temp)
        self.last_progress = float(progress)
        self.last_stage = stage_name

        # 随机打印监控 (1% 概率)

        if random.random() < 0.002:
            print(f"\n[SDT 监控 | 进度:{progress * 100:.1f}% | 阶段:{stage_name} | Temp:{temp:.3f}] "
                  f"W1:{weights[0, 0].item():.2f}, W2:{weights[0, 1].item():.2f}, "
                  f"W3:{weights[0, 2].item():.2f}, W4:{weights[0, 3].item():.2f}")


        return weights

# =========================================================================
# 🚀 SDT 路由权重通用净化与适配器 (拦截器)
# =========================================================================
def get_sdt_probs(sdt, num_experts=4, is_logits=False):
    """
    统一处理各种形式的 SDT 路由输出，强制净化为 [B, num_experts] 的软路由概率。
    sdt 允许的输入形式:
        1) [B, E] 概率分布 (如 SoftDecisionTree 的默认输出)
        2) [B, E] 未归一化的 logits
        3) [B] 强制指定的 expert id (常用于 Inference 测试阶段)
    return:
        sdt_probs: 绝对安全、无 NaN、归一化的 [B, E] 概率张量
    """
    if sdt is None:
        return None

    # 应对测试阶段强制指定专家 ID 的情况 (转为 one-hot)
    if sdt.dim() == 1:
        sdt_probs = F.one_hot(sdt.long(), num_classes=num_experts).float()
        return sdt_probs

    if sdt.dim() == 2:
        if is_logits:
            return F.softmax(sdt, dim=-1)
        else:
            sdt = sdt.clamp_min(1e-8)
            den = sdt.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return sdt / den

    raise ValueError(f"Unsupported SDT shape: {sdt.shape}")