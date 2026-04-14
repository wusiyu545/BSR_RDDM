import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1


class BFR_Stage2_Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        print("🚀 [Stage 2] 初始化高阶约束网络 (LPIPS + FaceID)...")
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
        self.id_fn = InceptionResnetV1(pretrained='vggface2').to(device).eval()

        for param in self.lpips_fn.parameters():
            param.requires_grad = False
        for param in self.id_fn.parameters():
            param.requires_grad = False

        # =========================================================
        # 🚀 身份损失增强模块：中心裁剪与标准化
        # =========================================================
        # 1. ImageNet 标准化 (LPIPS VGG 和多数特征网络通常喜欢这个分布)
        # 注意：扩散模型里的张量已经是 [-1, 1] 了，
        # 如果需要转到特定的 mean/std，可以在这里做可导的线性变换。
        # facenet-pytorch 默认期望 [-1, 1] 的输入，所以这里我们保持数值范围不变。

        # 2. 核心：动态中心裁剪 (CenterCrop)
        # FFHQ 包含很多背景。我们裁掉外围，只保留面部核心区域送入 FaceID，
        # 避免模型为了强行对齐背景和头发而扭曲人脸特征。
        # 假设原图是 512，裁剪系数设为 0.7 左右通常最佳。
        self.crop_ratio = 0.7

    def forward(self, pred_x0, target_x0):
        # 1. 计算 LPIPS (使用全尺寸图像，评估整体感知质量)
        loss_lpips = self.lpips_fn(pred_x0, target_x0)

        # 2. 计算 FaceID (剥离背景，只评估面部)
        b, c, h, w = pred_x0.shape
        crop_h, crop_w = int(h * self.crop_ratio), int(w * self.crop_ratio)

        # 使用可导的 T.CenterCrop 或者简单的切片操作
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2

        # 纯 Tensor 切片，绝对可导
        pred_face = pred_x0[:, :, start_y:start_y + crop_h, start_x:start_x + crop_w]
        target_face = target_x0[:, :, start_y:start_y + crop_h, start_x:start_x + crop_w]

        # 3. 插值到 160x160 (使用 anti_alias 抵抗高频锯齿变形)
        pred_id_in = F.interpolate(pred_face, size=(160, 160), mode='bicubic', align_corners=False, antialias=True)
        target_id_in = F.interpolate(target_face, size=(160, 160), mode='bicubic', align_corners=False, antialias=True)

        feat_pred = self.id_fn(pred_id_in)
        feat_target = self.id_fn(target_id_in)

        loss_id = 1.0 - F.cosine_similarity(feat_pred, feat_target, dim=-1)

        return loss_lpips, loss_id