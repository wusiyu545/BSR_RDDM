from pathlib import Path

BENCH_ROOT = Path("./data/benchmark")
OUT_ROOT = Path("./data/benchmark_flist")

DATASETS = ["Set5", "Set14", "BSDS100", "Urban100", "DF2KVal"]
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
    for name in DATASETS:
        hr_dir = BENCH_ROOT / name / "HR"
        imgs = collect_images(hr_dir)
        out_file = OUT_ROOT / f"{name}_hr.flist"
        write_flist(imgs, out_file)
        print(f"{name}: {len(imgs)} -> {out_file}")


if __name__ == "__main__":
    main()