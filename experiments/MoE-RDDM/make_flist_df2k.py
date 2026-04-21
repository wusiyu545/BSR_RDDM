from pathlib import Path

# =========================
# 路径配置
# =========================
ROOT = Path(r"./data/DF2K_HR")

DIV2K_DIR = ROOT / "DIV2K_HR"
FLICKR2K_DIR = ROOT / "Flickr2K_HR"

TRAIN_FLIST = ROOT / "train_hr.flist"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")

    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p.resolve())

    files = sorted(files, key=lambda x: str(x).lower())
    if not files:
        raise FileNotFoundError(f"目录中没有图片: {folder}")
    return files


def write_flist(paths, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for p in paths:
            # 统一写成正斜杠，避免 Windows 路径转义问题
            f.write(str(p).replace("\\", "/") + "\n")


def main():
    div2k_imgs = collect_images(DIV2K_DIR)
    flickr2k_imgs = collect_images(FLICKR2K_DIR)

    train_imgs = div2k_imgs + flickr2k_imgs
    train_imgs = sorted(train_imgs, key=lambda x: str(x).lower())

    write_flist(train_imgs, TRAIN_FLIST)

    print("DF2K train_hr.flist 生成完成：")
    print(f"  DIV2K_HR    : {len(div2k_imgs)}")
    print(f"  Flickr2K_HR : {len(flickr2k_imgs)}")
    print(f"  Total       : {len(train_imgs)}")
    print(f"  Output      : {TRAIN_FLIST}")


if __name__ == "__main__":
    main()