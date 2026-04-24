from pathlib import Path

ROOT = Path("./data/DF2K")
HR_DIR = ROOT / "HR"
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
            f.write(str(p).replace("\\", "/") + "\n")


def main():
    imgs = collect_images(HR_DIR)
    write_flist(imgs, TRAIN_FLIST)

    print("DF2K train_hr.flist 生成完成：")
    print(f"  HR images : {len(imgs)}")
    print(f"  Output    : {TRAIN_FLIST}")


if __name__ == "__main__":
    main()