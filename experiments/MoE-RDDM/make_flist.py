from pathlib import Path

# =========================
# 路径配置
# =========================
ROOT = Path(r"./data/DIV2K_x4_min")

TRAIN_DIR = ROOT / "train_HR_sub"
VAL_GT_DIR = ROOT / "val_HR"
VAL_INPUT_DIR = ROOT / "val_LQ_x4"

TRAIN_FLIST = ROOT / "train_gt.flist"
TEST_GT_FLIST = ROOT / "test_gt.flist"
TEST_INPUT_FLIST = ROOT / "test_input.flist"

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
    with open(out_file, "w", encoding="utf-8") as f:
        for p in paths:
            # 统一写成正斜杠，避免 Windows 路径转义问题
            f.write(str(p).replace("\\", "/") + "\n")


def main():
    train_imgs = collect_images(TRAIN_DIR)
    val_gt_imgs = collect_images(VAL_GT_DIR)
    val_input_imgs = collect_images(VAL_INPUT_DIR)

    write_flist(train_imgs, TRAIN_FLIST)
    write_flist(val_gt_imgs, TEST_GT_FLIST)
    write_flist(val_input_imgs, TEST_INPUT_FLIST)

    print("flist 生成完成：")
    print(f"  {TRAIN_FLIST.name}: {len(train_imgs)}")
    print(f"  {TEST_GT_FLIST.name}: {len(val_gt_imgs)}")
    print(f"  {TEST_INPUT_FLIST.name}: {len(val_input_imgs)}")


if __name__ == "__main__":
    main()