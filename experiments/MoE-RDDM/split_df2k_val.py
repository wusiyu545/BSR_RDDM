from pathlib import Path
import random
import shutil

# =========================
# 配置
# =========================
DF2K_HR_DIR = Path("./data/DF2K/HR")
VAL_DIR = Path("./data/benchmark/DF2KVal/HR")

TRAIN_FLIST = Path("./data/DF2K/train_hr.flist")
VAL_FLIST = Path("./data/benchmark_flist/DF2KVal_hr.flist")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

VAL_COUNT = 40
SEED = 42


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
    all_imgs = collect_images(DF2K_HR_DIR)
    if len(all_imgs) <= VAL_COUNT:
        raise RuntimeError(f"DF2K 总图像数 {len(all_imgs)} <= VAL_COUNT={VAL_COUNT}，无法划分。")

    rng = random.Random(SEED)
    val_imgs = rng.sample(all_imgs, VAL_COUNT)
    val_set = set(val_imgs)

    train_imgs = [p for p in all_imgs if p not in val_set]

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # 复制验证集到 benchmark/DF2KVal/HR
    for p in val_imgs:
        dst = VAL_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)

    # 写训练 / 验证 flist
    write_flist(train_imgs, TRAIN_FLIST)
    write_flist(sorted([p.resolve() for p in VAL_DIR.glob("*") if p.suffix.lower() in IMG_EXTS]), VAL_FLIST)

    print("DF2KVal 划分完成：")
    print(f"  Total DF2K : {len(all_imgs)}")
    print(f"  Train      : {len(train_imgs)}")
    print(f"  Val        : {VAL_COUNT}")
    print(f"  Train flist: {TRAIN_FLIST}")
    print(f"  Val flist  : {VAL_FLIST}")
    print(f"  Val dir    : {VAL_DIR}")


if __name__ == "__main__":
    main()