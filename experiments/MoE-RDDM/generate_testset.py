import cv2
import numpy as np
import os
from pathlib import Path


def get_standard_gaussian_kernel(kernel_size=15, sigma=2.0):
    """生成学术界标准的固定高斯模糊核"""
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def get_standard_motion_kernel(kernel_size=15, angle=45):
    """生成学术界标准的固定运动模糊核"""
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    # 简单的对角线运动模糊
    for i in range(kernel_size):
        offset = i - center
        x = center + int(offset * np.cos(np.radians(angle)))
        y = center + int(offset * np.sin(np.radians(angle)))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    return kernel / np.sum(kernel)


def add_standard_noise(img_bgr, sigma=0.01):
    """
    添加对齐标准盲逆实验的高斯噪声 (sigma=0.01)
    注意：图像的像素值范围是 0-255，所以 0.01 的噪声实际标准差是 0.01 * 255 = 2.55
    """
    noise = np.random.normal(0, sigma * 255.0, img_bgr.shape)
    img_noisy = img_bgr.astype(np.float32) + noise
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    return img_noisy


def generate_academic_testset():
    # 1. 路径设置
    # 请确保你的高清原图放在这个文件夹下
    gt_dir = 'data/test_images'
    out_gaussian_dir = './data/test_gaussian_blur'
    out_motion_dir = './data/test_motion_blur'

    Path(out_gaussian_dir).mkdir(parents=True, exist_ok=True)
    Path(out_motion_dir).mkdir(parents=True, exist_ok=True)

    # 2. 获取标准核
    kernel_gaussian = get_standard_gaussian_kernel(kernel_size=15, sigma=2.0)
    kernel_motion = get_standard_motion_kernel(kernel_size=15, angle=45)

    print("🚀 开始生成标准盲逆测试集...")
    count = 0
    for filename in os.listdir(gt_dir):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(gt_dir, filename)
        img_gt = cv2.imread(img_path)
        if img_gt is None:
            continue

        # --- 制作任务 A：高斯模糊 + 噪声的测试图 ---
        img_blur_g = cv2.filter2D(img_gt, -1, kernel_gaussian)
        img_final_g = add_standard_noise(img_blur_g, sigma=0.01)
        cv2.imwrite(os.path.join(out_gaussian_dir, filename), img_final_g)

        # --- 制作任务 B：运动模糊 + 噪声的测试图 ---
        img_blur_m = cv2.filter2D(img_gt, -1, kernel_motion)
        img_final_m = add_standard_noise(img_blur_m, sigma=0.01)
        cv2.imwrite(os.path.join(out_motion_dir, filename), img_final_m)

        count += 1

    print(f"✅ 成功生成！共处理了 {count} 张图片。")
    print(f"📁 高斯盲测试集保存在: {out_gaussian_dir}")
    print(f"📁 运动盲测试集保存在: {out_motion_dir}")


if __name__ == '__main__':
    generate_academic_testset()