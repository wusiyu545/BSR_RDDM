import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def random_blur(img, blur_prob=0.8, kernel_choices=None, sigma_min=0.2, sigma_max=3.0):
    """Apply random Gaussian blur to an RGB uint8 image."""
    if kernel_choices is None:
        kernel_choices = [7, 9, 11, 13, 15, 17, 19, 21]

    if random.random() < blur_prob:
        k = random.choice(kernel_choices)
        sigma = random.uniform(sigma_min, sigma_max)
        img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)

    return img


def random_noise(img, noise_prob=0.5, noise_min=0.0, noise_max=15.0):
    """Apply additive Gaussian noise to an RGB uint8 image."""
    if random.random() < noise_prob:
        sigma = random.uniform(noise_min, noise_max)
        noise = np.random.randn(*img.shape) * sigma
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def random_jpeg(img, jpeg_prob=0.7, jpeg_min=40, jpeg_max=95):
    """Apply random JPEG compression to an RGB uint8 image."""
    if random.random() < jpeg_prob:
        quality = random.randint(jpeg_min, jpeg_max)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def degrade_to_lr_and_lq_up(hr, args):
    """
    Generate LR and LQ_up from HR.

    HR and LQ_up have the same spatial size and are used by RDDM-BSR teacher:
        input  = LQ_up
        target = HR

    LR is also saved for the later LightBSR student stage.
    """
    scale = args.scale
    h, w = hr.shape[:2]
    h = h // scale * scale
    w = w // scale * scale
    hr = hr[:h, :w, :]

    img = random_blur(
        hr,
        blur_prob=args.blur_prob,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )

    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

    lr = random_noise(
        lr,
        noise_prob=args.noise_prob,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
    )

    lr = random_jpeg(
        lr,
        jpeg_prob=args.jpeg_prob,
        jpeg_min=args.jpeg_min,
        jpeg_max=args.jpeg_max,
    )

    lq_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

    return hr, lr, lq_up


def save_flist(paths, flist_path, append=False):
    mode = "a" if append else "w"
    with open(flist_path, mode, encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")


def build_name(index, name_prefix=""):
    if name_prefix:
        return f"{name_prefix}_{index:06d}.png"
    return f"{index:06d}.png"


def check_output_path(path, overwrite=False):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}\n"
            "Use a different --name_prefix / --start_index, or pass --overwrite if you intentionally want to replace it."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HR / LR / LQ_up pairs for RDDM-BSR teacher training."
    )

    parser.add_argument("--hr_root", type=str, required=True,
                        help="Root directory of HR images, e.g. DIV2K_train_HR or Flickr2K_HR.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for generated BSR pairs and flists.")
    parser.add_argument("--scale", type=int, default=4,
                        help="Super-resolution scale. For x4 SR, use 4.")
    parser.add_argument("--prefix", type=str, default="train",
                        help="Dataset split prefix, e.g. train, val, debug.")

    # Naming / merging control.
    parser.add_argument("--name_prefix", type=str, default="",
                        help="Prefix for output image names, e.g. div2k or flickr2k. Prevents file name conflicts when merging DF2K.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index for output file names.")
    parser.add_argument("--append_flist", action="store_true",
                        help="Append generated paths to existing flist files instead of overwriting them.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output images if they already exist.")
    parser.add_argument("--max_images", type=int, default=-1,
                        help="Maximum number of HR images to process. -1 means all images.")
    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed for degradation generation.")

    # Degradation parameters.
    parser.add_argument("--blur_prob", type=float, default=0.8,
                        help="Probability of applying Gaussian blur.")
    parser.add_argument("--sigma_min", type=float, default=0.2,
                        help="Minimum Gaussian blur sigma.")
    parser.add_argument("--sigma_max", type=float, default=3.0,
                        help="Maximum Gaussian blur sigma.")
    parser.add_argument("--noise_prob", type=float, default=0.5,
                        help="Probability of applying Gaussian noise.")
    parser.add_argument("--noise_min", type=float, default=0.0,
                        help="Minimum Gaussian noise sigma.")
    parser.add_argument("--noise_max", type=float, default=15.0,
                        help="Maximum Gaussian noise sigma.")
    parser.add_argument("--jpeg_prob", type=float, default=0.7,
                        help="Probability of applying JPEG compression.")
    parser.add_argument("--jpeg_min", type=int, default=40,
                        help="Minimum JPEG quality.")
    parser.add_argument("--jpeg_max", type=int, default=95,
                        help="Maximum JPEG quality.")

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    hr_root = Path(args.hr_root)
    out_dir = Path(args.out_dir)

    hr_out = out_dir / f"{args.prefix}_hr"
    lr_out = out_dir / f"{args.prefix}_lr"
    lq_up_out = out_dir / f"{args.prefix}_lq_up"

    hr_out.mkdir(parents=True, exist_ok=True)
    lr_out.mkdir(parents=True, exist_ok=True)
    lq_up_out.mkdir(parents=True, exist_ok=True)

    hr_paths = []
    lr_paths = []
    lq_up_paths = []

    images = list_images(hr_root)
    if args.max_images > 0:
        images = images[:args.max_images]

    if len(images) == 0:
        raise RuntimeError(f"No images found in {hr_root}")

    for local_i, img_path in enumerate(images):
        global_i = args.start_index + local_i
        name = build_name(global_i, args.name_prefix)

        hr_save = hr_out / name
        lr_save = lr_out / name
        lq_up_save = lq_up_out / name

        check_output_path(hr_save, overwrite=args.overwrite)
        check_output_path(lr_save, overwrite=args.overwrite)
        check_output_path(lq_up_save, overwrite=args.overwrite)

        img = Image.open(img_path).convert("RGB")
        hr = np.array(img)

        hr, lr, lq_up = degrade_to_lr_and_lq_up(hr, args)

        Image.fromarray(hr).save(hr_save)
        Image.fromarray(lr).save(lr_save)
        Image.fromarray(lq_up).save(lq_up_save)

        hr_paths.append(hr_save.resolve())
        lr_paths.append(lr_save.resolve())
        lq_up_paths.append(lq_up_save.resolve())

        if (local_i + 1) % 100 == 0 or (local_i + 1) == len(images):
            print(f"Processed {local_i + 1}/{len(images)} images")

    save_flist(hr_paths, out_dir / f"{args.prefix}_hr.flist", append=args.append_flist)
    save_flist(lr_paths, out_dir / f"{args.prefix}_lr.flist", append=args.append_flist)
    save_flist(lq_up_paths, out_dir / f"{args.prefix}_lq_up.flist", append=args.append_flist)

    mode = "appended to" if args.append_flist else "written to"
    print(f"Generated {len(images)} pairs at {out_dir}")
    print(f"Flist files have been {mode}: {out_dir}")


if __name__ == "__main__":
    main()
