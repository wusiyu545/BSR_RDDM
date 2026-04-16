from pathlib import Path
import random
import cv2
import numpy as np

from my_datasets.base import _apply_jpeg, _sample_blur_kernel

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data" / "DIV2K_x4_min"
SRC_DIR = ROOT / "val_HR"
DST_DIR = ROOT / "val_LQ_x4_blind"

DST_DIR.mkdir(parents=True, exist_ok=True)


def degrade_fixed_blind_sr_x4(img_gt, seed, sr_scale=4, two_order_deg=True):
    # 用固定 seed 保证每张验证图退化是可复现的
    random.seed(seed)
    np.random.seed(seed)

    img = img_gt.astype(np.float32)
    h, w = img.shape[:2]

    # 1) first-order blur
    if random.random() < 0.9:
        kernel, _, _ = _sample_blur_kernel()
        img = cv2.filter2D(img, -1, kernel)

    # 2) mild pre-resize perturbation
    if random.random() < 0.5:
        resize_factor = random.uniform(0.95, 1.05)
        inter_mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
        tmp_w = max(1, int(round(w * resize_factor)))
        tmp_h = max(1, int(round(h * resize_factor)))
        img = cv2.resize(img, (tmp_w, tmp_h), interpolation=inter_mode)

    # 3) force native x4 LR
    lr_w = max(1, w // sr_scale)
    lr_h = max(1, h // sr_scale)
    inter_mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    img = cv2.resize(img, (lr_w, lr_h), interpolation=inter_mode)

    # 4) noise on LR
    if random.random() < 0.8:
        noise_level = random.uniform(1.0, 12.0)
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        img = img + noise

    # 5) JPEG on LR
    img = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.8:
        q = random.randint(30, 95)
        img = _apply_jpeg(img, q)

    # 6) optional second-order light degradation
    if two_order_deg and random.random() < 0.5:
        # second blur
        if random.random() < 0.5:
            kernel2, _, _ = _sample_blur_kernel()
            img = cv2.filter2D(img, -1, kernel2)

        # second noise
        if random.random() < 0.5:
            noise_level2 = random.uniform(0.5, 5.0)
            noise2 = np.random.normal(0, noise_level2, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise2, 0, 255).astype(np.uint8)

        # second JPEG
        if random.random() < 0.5:
            q2 = random.randint(50, 95)
            img = _apply_jpeg(img, q2)

    # 7) upsample back to GT size
    up_mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    img_lq_up = cv2.resize(img, (w, h), interpolation=up_mode)
    img_lq_up = np.ascontiguousarray(img_lq_up)

    return img_lq_up


def main():
    files = sorted(list(SRC_DIR.glob("*.png")) + list(SRC_DIR.glob("*.jpg")) + list(SRC_DIR.glob("*.jpeg")))
    print(f"[INFO] found {len(files)} HR validation images.")

    for idx, path in enumerate(files):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] skip unreadable image: {path}")
            continue

        # 固定 seed：同一张图每次都生成同一个 blind 输入
        seed = 2026 + idx

        out = degrade_fixed_blind_sr_x4(img, seed=seed, sr_scale=4, two_order_deg=True)
        out_path = DST_DIR / path.name
        cv2.imwrite(str(out_path), out)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(files):
            print(f"[INFO] processed {idx + 1}/{len(files)}")

    print(f"[DONE] saved fixed blind validation inputs to: {DST_DIR}")


if __name__ == "__main__":
    main()