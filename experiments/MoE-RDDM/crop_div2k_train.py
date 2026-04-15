from pathlib import Path
from PIL import Image

# =========================
# 路径配置
# =========================
ROOT = Path(r"./data/DIV2K_x4_min")
SRC_DIR = ROOT / "train_HR"
DST_DIR = ROOT / "train_HR_sub"

# =========================
# 裁图参数
# =========================
CROP_SIZE = 256
STRIDE = 128

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in SRC_DIR.iterdir() if is_image_file(p)])
    if not img_paths:
        raise FileNotFoundError(f"未在 {SRC_DIR} 中找到图片文件。")

    total_saved = 0

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        stem = img_path.stem
        idx = 1

        # 正常滑窗裁图
        if w >= CROP_SIZE and h >= CROP_SIZE:
            y_positions = list(range(0, h - CROP_SIZE + 1, STRIDE))
            x_positions = list(range(0, w - CROP_SIZE + 1, STRIDE))

            # 保证覆盖到右边界和下边界
            if y_positions[-1] != h - CROP_SIZE:
                y_positions.append(h - CROP_SIZE)
            if x_positions[-1] != w - CROP_SIZE:
                x_positions.append(w - CROP_SIZE)

            for top in y_positions:
                for left in x_positions:
                    patch = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
                    out_path = DST_DIR / f"{stem}_s{idx:03d}.png"
                    patch.save(out_path)
                    idx += 1
                    total_saved += 1
        else:
            # 图片太小，保底缩放一张
            patch = img.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
            out_path = DST_DIR / f"{stem}_s001.png"
            patch.save(out_path)
            total_saved += 1

    print(f"完成。共保存 {total_saved} 张子图到: {DST_DIR}")


if __name__ == "__main__":
    main()