import argparse
import subprocess
import sys
from pathlib import Path


"""
Prepare validation and test sets for RDDM-BSR teacher.

This script is a small wrapper around generate_bsr_pairs.py. It creates the
same paired file structure for validation and test splits:

out_dir/
├── val_hr/
├── val_lr/
├── val_lq_up/
├── val_hr.flist
├── val_lr.flist
├── val_lq_up.flist
├── test_hr/
├── test_lr/
├── test_lq_up/
├── test_hr.flist
├── test_lr.flist
└── test_lq_up.flist

For RDDM teacher training, train.py uses:
    --val_hr out_dir/val_hr.flist
    --val_lq_up out_dir/val_lq_up.flist

For later teacher-output export or LightBSR distillation, the LR flist can also
be used:
    --lr_flist out_dir/val_lr.flist
    --lr_flist out_dir/test_lr.flist
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare validation and test HR/LR/LQ_up pairs for RDDM-BSR teacher."
    )

    parser.add_argument("--val_hr_root", type=str, default=None,
                        help="Root directory of validation HR images. If omitted, validation set is skipped.")
    parser.add_argument("--test_hr_root", type=str, default=None,
                        help="Root directory of test HR images. If omitted, test set is skipped.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for generated val/test pairs and flists.")
    parser.add_argument("--scale", type=int, default=4,
                        help="Super-resolution scale, usually 4.")

    parser.add_argument("--val_name_prefix", type=str, default="val",
                        help="Image name prefix for validation images.")
    parser.add_argument("--test_name_prefix", type=str, default="test",
                        help="Image name prefix for test images.")
    parser.add_argument("--max_val_images", type=int, default=-1,
                        help="Maximum validation images to process. -1 means all.")
    parser.add_argument("--max_test_images", type=int, default=-1,
                        help="Maximum test images to process. -1 means all.")
    parser.add_argument("--seed", type=int, default=10,
                        help="Base random seed. Validation uses seed, test uses seed + 100.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output images.")

    # Keep degradation parameters consistent with generate_bsr_pairs.py.
    parser.add_argument("--blur_prob", type=float, default=0.8)
    parser.add_argument("--sigma_min", type=float, default=0.2)
    parser.add_argument("--sigma_max", type=float, default=3.0)
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--noise_min", type=float, default=0.0)
    parser.add_argument("--noise_max", type=float, default=15.0)
    parser.add_argument("--jpeg_prob", type=float, default=0.7)
    parser.add_argument("--jpeg_min", type=int, default=40)
    parser.add_argument("--jpeg_max", type=int, default=95)

    return parser.parse_args()


def run_generate(split, hr_root, out_dir, scale, name_prefix, max_images, seed, args):
    if hr_root is None:
        print(f"[skip] {split}: hr_root is not provided")
        return

    hr_root = Path(hr_root)
    if not hr_root.exists():
        raise FileNotFoundError(f"{split} HR root does not exist: {hr_root}")

    script_path = Path(__file__).resolve().parent / "generate_bsr_pairs.py"
    if not script_path.exists():
        raise FileNotFoundError(f"generate_bsr_pairs.py not found: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--hr_root", str(hr_root),
        "--out_dir", str(out_dir),
        "--scale", str(scale),
        "--prefix", split,
        "--name_prefix", name_prefix,
        "--seed", str(seed),
        "--blur_prob", str(args.blur_prob),
        "--sigma_min", str(args.sigma_min),
        "--sigma_max", str(args.sigma_max),
        "--noise_prob", str(args.noise_prob),
        "--noise_min", str(args.noise_min),
        "--noise_max", str(args.noise_max),
        "--jpeg_prob", str(args.jpeg_prob),
        "--jpeg_min", str(args.jpeg_min),
        "--jpeg_max", str(args.jpeg_max),
    ]

    if max_images > 0:
        cmd.extend(["--max_images", str(max_images)])

    if args.overwrite:
        cmd.append("--overwrite")

    print("\n[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_generate(
        split="val",
        hr_root=args.val_hr_root,
        out_dir=out_dir,
        scale=args.scale,
        name_prefix=args.val_name_prefix,
        max_images=args.max_val_images,
        seed=args.seed,
        args=args,
    )

    run_generate(
        split="test",
        hr_root=args.test_hr_root,
        out_dir=out_dir,
        scale=args.scale,
        name_prefix=args.test_name_prefix,
        max_images=args.max_test_images,
        seed=args.seed + 100,
        args=args,
    )

    print("\nDone. Expected files:")
    for split in ["val", "test"]:
        print(f"  {out_dir / (split + '_hr.flist')}")
        print(f"  {out_dir / (split + '_lr.flist')}")
        print(f"  {out_dir / (split + '_lq_up.flist')}")


if __name__ == "__main__":
    main()
