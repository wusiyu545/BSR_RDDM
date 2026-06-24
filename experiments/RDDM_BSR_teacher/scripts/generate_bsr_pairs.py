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


def random_blur(img):
    # img: RGB, uint8
    if random.random() < 0.8:
        k = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
        sigma = random.uniform(0.2, 3.0)
        img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    return img


def random_noise(img):
    if random.random() < 0.5:
        sigma = random.uniform(0, 15)
        noise = np.random.randn(*img.shape) * sigma
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def random_jpeg(img):
    if random.random() < 0.7:
        quality = random.randint(40, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def degrade_to_lr_and_lq_up(hr, scale=4):
    h, w = hr.shape[:2]
    h = h // scale * scale
    w = w // scale * scale
    hr = hr[:h, :w, :]

    img = random_blur(hr)
    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    lr = random_noise(lr)
    lr = random_jpeg(lr)
    lq_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

    return hr, lr, lq_up


def save_flist(paths, flist_path):
    with open(flist_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--prefix", type=str, default="train")
    args = parser.parse_args()

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

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        hr = np.array(img)

        hr, lr, lq_up = degrade_to_lr_and_lq_up(hr, scale=args.scale)

        name = f"{i:06d}.png"

        hr_save = hr_out / name
        lr_save = lr_out / name
        lq_up_save = lq_up_out / name

        Image.fromarray(hr).save(hr_save)
        Image.fromarray(lr).save(lr_save)
        Image.fromarray(lq_up).save(lq_up_save)

        hr_paths.append(hr_save.resolve())
        lr_paths.append(lr_save.resolve())
        lq_up_paths.append(lq_up_save.resolve())

    save_flist(hr_paths, out_dir / f"{args.prefix}_hr.flist")
    save_flist(lr_paths, out_dir / f"{args.prefix}_lr.flist")
    save_flist(lq_up_paths, out_dir / f"{args.prefix}_lq_up.flist")

    print(f"Generated {len(images)} pairs at {out_dir}")


if __name__ == "__main__":
    main()