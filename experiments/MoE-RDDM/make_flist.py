import os
import glob
import cv2
import numpy as np
from pathlib import Path


# ==========================================
# 配置区
# ==========================================
BASE_DIR = "./data/FFHQ_512"
TRAIN_DIR = os.path.join(BASE_DIR, "train_gt")
TEST_GT_DIR = os.path.join(BASE_DIR, "test_gt")
TEST_INPUT_DIR = os.path.join(BASE_DIR, "test_input")

TRAIN_FLIST = os.path.join(BASE_DIR, "train_gt.flist")
TEST_GT_FLIST = os.path.join(BASE_DIR, "test_gt.flist")
TEST_INPUT_FLIST = os.path.join(BASE_DIR, "test_input.flist")

IMAGE_SIZE = 512

# 是否把 test_gt 中非 512 的图片覆盖改写成 512
OVERWRITE_TEST_GT = False

# 固定随机种子，保证 test_input 可复现
RANDOM_SEED = 1234


# ==========================================
# 退化算子
# ==========================================
def get_standard_gaussian_kernel(kernel_size=15, sigma=2.0):
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel_sum = np.sum(kernel)
    if kernel_sum <= 0:
        raise ValueError("Gaussian kernel sum is non-positive.")
    return kernel / kernel_sum


def get_standard_motion_kernel(kernel_size=15, angle=45):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    for i in range(kernel_size):
        offset = i - center
        x = center + int(round(offset * np.cos(np.radians(angle))))
        y = center + int(round(offset * np.sin(np.radians(angle))))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    kernel_sum = np.sum(kernel)
    if kernel_sum <= 0:
        raise ValueError("Motion kernel sum is non-positive.")
    return kernel / kernel_sum


# ==========================================
# 工具函数
# ==========================================
def is_image_file(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def list_images_sorted(folder: str):
    files = glob.glob(os.path.join(folder, "*"))
    files = [f for f in files if os.path.isfile(f) and is_image_file(f)]
    return sorted(files)


def to_abs_posix(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def resize_if_needed(img, size=512):
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# ==========================================
# BFR 混合退化
# ==========================================
def degrade_mixed_bfr(img):
    """
    更贴近 blind face restoration 的混合退化：
    Blur + DownUp + Noise + JPEG
    每张图随机采样一组退化参数
    """
    img = img.astype(np.float32)

    # 1) Blur
    if np.random.rand() < 0.8:
        if np.random.rand() < 0.7:
            k = int(np.random.choice([13, 17, 21, 27, 33]))
            sigma = float(np.random.uniform(3.0, 8.0))
            kernel = get_standard_gaussian_kernel(k, sigma)
            blur_tag = f"Gaussian(k={k},sigma={sigma:.2f})"
        else:
            k = int(np.random.choice([15, 21, 27, 33]))
            angle = float(np.random.uniform(0, 180))
            kernel = get_standard_motion_kernel(k, angle)
            blur_tag = f"Motion(k={k},angle={angle:.1f})"
        img = cv2.filter2D(img, -1, kernel)
    else:
        blur_tag = "NoBlur"

    # 2) Downsample / Upsample
    if np.random.rand() < 0.7:
        h, w = img.shape[:2]
        scale = float(np.random.uniform(0.125, 0.6))
        down_w = max(1, int(w * scale))
        down_h = max(1, int(h * scale))
        img = cv2.resize(img, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        scale_tag = f"DownUp(scale={scale:.3f})"
    else:
        scale_tag = "NoDownUp"

    # 3) Gaussian noise
    if np.random.rand() < 0.8:
        noise_level = float(np.random.uniform(2, 20))
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        img = img + noise
        noise_tag = f"Noise(std={noise_level:.2f})"
    else:
        noise_tag = "NoNoise"

    img = np.clip(img, 0, 255).astype(np.uint8)

    # 4) JPEG compression
    if np.random.rand() < 0.8:
        q = int(np.random.randint(30, 96))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, 1)
        jpeg_tag = f"JPEG(q={q})"
    else:
        jpeg_tag = "NoJPEG"

    tag = f"{blur_tag}, {scale_tag}, {noise_tag}, {jpeg_tag}"
    return img, tag


# ==========================================
# 主流程
# ==========================================
def main():
    np.random.seed(RANDOM_SEED)
    ensure_dir(TEST_INPUT_DIR)

    # 1) 生成 train_gt.flist
    train_images = list_images_sorted(TRAIN_DIR)
    if len(train_images) == 0:
        raise FileNotFoundError(f"训练目录为空或不存在有效图片: {TRAIN_DIR}")

    with open(TRAIN_FLIST, "w", encoding="utf-8") as f:
        for img_path in train_images:
            f.write(to_abs_posix(img_path) + "\n")

    print(f"✅ 成功生成 train_gt.flist，记录了 {len(train_images)} 张训练图片")

    # 2) 扫描 test_gt 并生成 test_input
    test_images = list_images_sorted(TEST_GT_DIR)
    if len(test_images) == 0:
        raise FileNotFoundError(f"测试 GT 目录为空或不存在有效图片: {TEST_GT_DIR}")

    clean_paths = []
    input_paths = []

    print(f"\n🚀 开始读取 {len(test_images)} 张 test_gt 图片，并生成混合退化 test_input...")

    for img_path in test_images:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ 跳过损坏图片: {img_path}")
            continue

        img = resize_if_needed(img, IMAGE_SIZE)

        if OVERWRITE_TEST_GT:
            cv2.imwrite(img_path, img)

        img_degraded, degrade_tag = degrade_mixed_bfr(img)

        filename = os.path.basename(img_path)
        degraded_path = os.path.join(TEST_INPUT_DIR, filename)
        cv2.imwrite(degraded_path, img_degraded)

        clean_paths.append(to_abs_posix(img_path))
        input_paths.append(to_abs_posix(degraded_path))

        print(f"  - 已处理 {filename} -> {degrade_tag}")

    if len(clean_paths) == 0 or len(input_paths) == 0:
        raise RuntimeError("没有成功生成任何测试样本，请检查 test_gt 中的图片。")

    # 3) 写 flist
    with open(TEST_GT_FLIST, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_paths) + "\n")
    print(f"\n✅ 成功生成 test_gt.flist，记录了 {len(clean_paths)} 张干净测试图")

    with open(TEST_INPUT_FLIST, "w", encoding="utf-8") as f:
        f.write("\n".join(input_paths) + "\n")
    print(f"✅ 成功生成 test_input.flist，记录了 {len(input_paths)} 张混合退化测试图")

    print(f"📁 混合退化图已保存在: {TEST_INPUT_DIR}")
    print("✅ 全部完成")


if __name__ == "__main__":
    main()