from pathlib import Path
from PIL import Image

# =========================
# 路径配置
# =========================
ROOT = Path(r"./data/DIV2K_x4_min")
SRC_DIR = ROOT / "val_HR"
DST_DIR = ROOT / "val_LQ_x4"

# =========================
# 超分倍率
# =========================
SCALE = 4

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in SRC_DIR.iterdir() if is_image_file(p)])
    if not img_paths:
        raise FileNotFoundError(f"未在 {SRC_DIR} 中找到图片文件。")

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        lr_w = max(1, w // SCALE)
        lr_h = max(1, h // SCALE)

        # 先降采样成原生 LR
        lr = img.resize((lr_w, lr_h), Image.BICUBIC)

        # 再升回 HR 尺寸，兼容当前仓库
        lq_up = lr.resize((w, h), Image.BICUBIC)

        out_path = DST_DIR / img_path.name
        lq_up.save(out_path)

    print(f"完成。已生成验证输入到: {DST_DIR}")


if __name__ == "__main__":
    main()