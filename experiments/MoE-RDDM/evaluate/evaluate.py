import cv2
import os
import torch
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 🌟 评估配置区 (与 test.py 保持一致)
# ==========================================
task_name = "Gaussian_Blur"  # 切换你想评估的任务: "Gaussian_Blur" 或 "Motion_Blur"
sampling_step = 250  # 你测试时设定的步数

# 路径设置
gt_dir = './data/source_images'  # 你的高清原图文件夹
restored_dir = f'./results/test_{task_name}_step{sampling_step}'  # 自动拼出测试结果文件夹
# ==========================================

# 初始化设备 (优先使用 GPU 以加速 LPIPS 特征提取)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用设备: {device}")

# 初始化 LPIPS 模型 (首次运行会自动下载 VGG 权重文件)
print("正在加载 LPIPS(VGG) 模型...")
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

total_psnr = 0
total_ssim = 0
total_lpips = 0
count = 0

print(f"\n🚀 开始测算 {task_name} 任务的综合评估指标 (PSNR / SSIM / LPIPS)...")
print(f"📁 正在读取修复结果: {restored_dir}\n")


def img2tensor_for_lpips(img_bgr):
    """将 OpenCV 读取的 BGR 图像转换为 LPIPS 所需的张量格式"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor.unsqueeze(0).to(device)


# 遍历修复图文件夹
for rest_filename in os.listdir(restored_dir):
    if not rest_filename.endswith(('.jpg', '.png', '.jpeg')):
        continue


    gt_filename = rest_filename

    gt_path = os.path.join(gt_dir, gt_filename)
    rest_path = os.path.join(restored_dir, rest_filename)

    # 兼容性处理：如果原图是 .jpg，但修复结果存成了 .png
    if not os.path.exists(gt_path):
        base_name = os.path.splitext(rest_filename)[0]
        # 尝试寻找 .jpg 或 .png
        if os.path.exists(os.path.join(gt_dir, base_name + '.jpg')):
            gt_path = os.path.join(gt_dir, base_name + '.jpg')
        elif os.path.exists(os.path.join(gt_dir, base_name + '.png')):
            gt_path = os.path.join(gt_dir, base_name + '.png')
        else:
            print(f"⚠️ 找不到对应的原图: {gt_filename}，跳过...")
            continue

    # 读取图像
    img_gt = cv2.imread(gt_path)
    img_rest = cv2.imread(rest_path)

    # 尺寸对齐 (防范边缘裁剪带来的微小误差)
    if img_gt.shape != img_rest.shape:
        img_rest = cv2.resize(img_rest, (img_gt.shape[1], img_gt.shape[0]))

    # --- 1. 计算 PSNR 和 SSIM ---
    val_psnr = psnr(img_gt, img_rest, data_range=255)
    val_ssim = ssim(img_gt, img_rest, data_range=255, channel_axis=-1)

    # --- 2. 计算 LPIPS ---
    tensor_gt = img2tensor_for_lpips(img_gt)
    tensor_rest = img2tensor_for_lpips(img_rest)

    with torch.no_grad():
        val_lpips = loss_fn_vgg(tensor_gt, tensor_rest).item()

    total_psnr += val_psnr
    total_ssim += val_ssim
    total_lpips += val_lpips
    count += 1

    print(f"[{rest_filename}] PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f}")

# 输出总成绩
if count > 0:
    print(f"\n🎉 === {task_name} 测试集 ({count}张) 平均表现 ===")
    print(f"平均 PSNR  (↑越大越好): {total_psnr / count:.2f} dB")
    print(f"平均 SSIM  (↑越大越好): {total_ssim / count:.4f}")
    print(f"平均 LPIPS (↓越小越好): {total_lpips / count:.4f}")
else:
    print("\n❌ 没有找到任何可以匹配的图片，请检查文件夹路径！")