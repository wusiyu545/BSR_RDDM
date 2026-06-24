import argparse
import os
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import utils
from PIL import Image

from datasets.base import Dataset
from src.residual_denoising_diffusion_pytorch import (
    ResidualDiffusion,
    Trainer,
    UnetRes,
    set_seed,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SR_teacher and R_teacher from a trained RDDM-BSR teacher."
    )

    # Input paired data.
    # HR and LQ_up must have the same spatial size.
    parser.add_argument("--hr_flist", type=str, required=True,
                        help="Path to HR flist.")
    parser.add_argument("--lq_up_flist", type=str, required=True,
                        help="Path to LQ_up flist. LQ_up is LR bicubic-upsampled to HR size.")
    parser.add_argument("--lr_flist", type=str, default=None,
                        help="Optional LR flist. If provided, LR images will be saved/copied to out_dir/lr.")

    # Checkpoint.
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model-{milestone}.pt.")
    parser.add_argument("--milestone", type=int, required=True,
                        help="Checkpoint milestone, e.g. 120 means model-120.pt.")

    # Output.
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save teacher outputs.")
    parser.add_argument("--keep_original_name", action="store_true",
                        help="Use original image basename. If not set, index prefix is added to avoid name collisions.")
    parser.add_argument("--save_residual_png", action="store_true",
                        help="Also save a visualized residual PNG. The training residual is always saved as .pt.")

    # Export mode.
    parser.add_argument("--export_mode", type=str, default="full", choices=["full", "patch"],
                        help="full: export whole images. patch: randomly crop patches before RDDM sampling.")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="HR/LQ_up patch size for --export_mode patch.")
    parser.add_argument("--patches_per_image", type=int, default=4,
                        help="Number of random patches exported from each image in patch mode.")
    parser.add_argument("--scale", type=int, default=4,
                        help="SR scale. Used to crop aligned LR patches in patch mode.")

    # Runtime.
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES setting.")
    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size. Keep 1 for large images or patch export.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start image index for exporting.")
    parser.add_argument("--max_items", type=int, default=-1,
                        help="Maximum number of exported samples. In patch mode this counts patches; in full mode this counts images. -1 means all.")
    parser.add_argument("--amp", action="store_true",
                        help="Use autocast during teacher sampling to reduce memory usage.")

    # Model settings. These should match the training script.
    parser.add_argument("--image_size", type=int, default=256,
                        help="Minimum image size / training patch size used by Dataset padding.")
    parser.add_argument("--dim", type=int, default=64,
                        help="Base U-Net channel dimension.")
    parser.add_argument("--share_encoder", type=int, default=0,
                        choices=[-1, 0, 1],
                        help="UnetRes share_encoder setting. Must match training.")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Diffusion timesteps. Must match training.")
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Sampling timesteps for DDIM inference.")
    parser.add_argument("--sum_scale", type=float, default=1.0,
                        help="RDDM sum_scale. Must match training.")
    parser.add_argument("--objective", type=str, default="pred_res_noise",
                        choices=[
                            "pred_res_noise",
                            "pred_res_add_noise",
                            "pred_x0_noise",
                            "pred_x0_add_noise",
                            "pred_noise",
                            "pred_res",
                        ],
                        help="RDDM objective. Must match training.")
    parser.add_argument("--loss_type", type=str, default="l1",
                        choices=["l1", "l2"],
                        help="Loss type. Must match training.")

    return parser.parse_args()


def read_flist(path):
    if path is None:
        return None

    path = Path(path)
    if path.is_file() and path.suffix.lower() not in IMAGE_EXTS:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    if path.is_dir():
        return sorted([str(p) for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS])

    if path.is_file():
        return [str(path)]

    return None


def build_diffusion(args):
    condition = True
    input_condition = False
    input_condition_mask = False

    model = UnetRes(
        dim=args.dim,
        dim_mults=(1, 2, 4, 8),
        share_encoder=args.share_encoder,
        condition=condition,
        input_condition=input_condition,
    )

    diffusion = ResidualDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
        loss_type=args.loss_type,
        condition=condition,
        sum_scale=args.sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
    )

    return diffusion, condition


def ensure_dirs(out_dir):
    out_dir = Path(out_dir)
    dirs = {
        "hr": out_dir / "hr",
        "lq_up": out_dir / "lq_up",
        "sr_teacher": out_dir / "sr_teacher",
        "residual": out_dir / "residual",
        "lr": out_dir / "lr",
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    return dirs


def make_name(dataset, index, keep_original_name=False):
    try:
        base_name = dataset.load_name(index, sub_dir=0)
    except Exception:
        base_name = f"{index:06d}.png"

    base_name = os.path.basename(str(base_name))
    stem = Path(base_name).stem

    if keep_original_name:
        return stem + ".png"

    return f"{index:06d}_{stem}.png"


def make_patch_name(dataset, image_index, patch_index, keep_original_name=False):
    base_name = make_name(dataset, image_index, keep_original_name=keep_original_name)
    stem = Path(base_name).stem
    return f"{stem}_p{patch_index:03d}.png"


def crop_pad(tensor, pad_size):
    """
    tensor: [B, C, H, W]
    pad_size: [bottom, right]
    """
    bottom, right = pad_size
    if bottom == 0 and right == 0:
        return tensor

    _, _, h, w = tensor.shape
    h_end = h - bottom if bottom > 0 else h
    w_end = w - right if right > 0 else w
    return tensor[:, :, :h_end, :w_end]


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(tensor.clamp(0.0, 1.0), str(path), nrow=1)


def copy_or_save_lr(lr_paths, index, dst_path):
    if lr_paths is None or index >= len(lr_paths):
        return

    src = Path(lr_paths[index])
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if src.exists():
        try:
            img = Image.open(src).convert("RGB")
            img.save(dst_path)
        except Exception:
            shutil.copyfile(src, dst_path)


def save_lr_patch(lr_paths, image_index, rr, cc, patch_size, scale, dst_path):
    if lr_paths is None or image_index >= len(lr_paths):
        return

    src = Path(lr_paths[image_index])
    if not src.exists():
        return

    lr_patch_size = patch_size // scale
    if lr_patch_size <= 0:
        return

    lr_rr = rr // scale
    lr_cc = cc // scale

    img = Image.open(src).convert("RGB")
    w, h = img.size

    # Make sure the LR crop stays in range. This matters when HR dimensions are not exactly scale-aligned.
    lr_rr = min(max(lr_rr, 0), max(h - lr_patch_size, 0))
    lr_cc = min(max(lr_cc, 0), max(w - lr_patch_size, 0))

    patch = img.crop((lr_cc, lr_rr, lr_cc + lr_patch_size, lr_rr + lr_patch_size))
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(dst_path)


def sample_rddm(trainer, lq_up, use_amp=False):
    """Run RDDM teacher sampling on a LQ_up tensor [B, C, H, W]."""
    amp_enabled = bool(use_amp and torch.cuda.is_available())

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        samples = list(
            trainer.ema.ema_model.sample(
                [lq_up],
                batch_size=lq_up.shape[0],
                last=True,
            )
        )

    return samples[-1].clamp(0.0, 1.0)


def save_teacher_sample(hr_b, lq_b, sr_b, out_dirs, name, save_residual_png=False):
    r_teacher = sr_b - lq_b

    save_tensor_image(hr_b, out_dirs["hr"] / name)
    save_tensor_image(lq_b, out_dirs["lq_up"] / name)
    save_tensor_image(sr_b, out_dirs["sr_teacher"] / name)

    residual_path = out_dirs["residual"] / (Path(name).stem + ".pt")
    torch.save(r_teacher.detach().cpu(), residual_path)

    if save_residual_png:
        # Visualization only: map roughly from [-1,1] to [0,1].
        residual_vis = (r_teacher + 1.0) * 0.5
        save_tensor_image(residual_vis, out_dirs["residual"] / name)


def export_full(args, trainer, loader, dataset, lr_paths, out_dirs, device):
    exported = 0

    with torch.no_grad():
        for batch_index, items in enumerate(loader):
            for b in range(items[0].shape[0]):
                image_index = batch_index * args.batch_size + b
                if image_index < args.start_index:
                    continue
                if args.max_items > 0 and exported >= args.max_items:
                    return exported

                hr_b = items[0][b:b + 1].to(device)
                lq_b = items[1][b:b + 1].to(device)

                sr_b = sample_rddm(trainer, lq_b, use_amp=args.amp)

                # Remove padding added by Dataset.pad_img.
                try:
                    pad_size = dataset.get_pad_size(image_index)
                    hr_b = crop_pad(hr_b, pad_size)
                    lq_b = crop_pad(lq_b, pad_size)
                    sr_b = crop_pad(sr_b, pad_size)
                except Exception:
                    pass

                name = make_name(dataset, image_index, args.keep_original_name)
                save_teacher_sample(
                    hr_b,
                    lq_b,
                    sr_b,
                    out_dirs,
                    name,
                    save_residual_png=args.save_residual_png,
                )
                copy_or_save_lr(lr_paths, image_index, out_dirs["lr"] / name)

                exported += 1
                print(f"[{exported}] saved full teacher output: {name}")

    return exported


def export_patch(args, trainer, loader, dataset, lr_paths, out_dirs, device):
    if args.patch_size % args.scale != 0:
        raise ValueError(
            f"patch_size must be divisible by scale for aligned LR patches, "
            f"but got patch_size={args.patch_size}, scale={args.scale}."
        )

    exported = 0

    with torch.no_grad():
        for batch_index, items in enumerate(loader):
            for b in range(items[0].shape[0]):
                image_index = batch_index * args.batch_size + b
                if image_index < args.start_index:
                    continue

                hr_full = items[0][b:b + 1]
                lq_full = items[1][b:b + 1]

                # Remove padding before cropping patches so saved patches correspond to real image areas.
                try:
                    pad_size = dataset.get_pad_size(image_index)
                    hr_full = crop_pad(hr_full, pad_size)
                    lq_full = crop_pad(lq_full, pad_size)
                except Exception:
                    pass

                _, _, h, w = hr_full.shape
                if h < args.patch_size or w < args.patch_size:
                    print(
                        f"[skip] image_index={image_index} is smaller than patch_size: "
                        f"image=({h}, {w}), patch_size={args.patch_size}"
                    )
                    continue

                for patch_index in range(args.patches_per_image):
                    if args.max_items > 0 and exported >= args.max_items:
                        return exported

                    rr = random.randint(0, h - args.patch_size)
                    cc = random.randint(0, w - args.patch_size)

                    hr_patch = hr_full[:, :, rr:rr + args.patch_size, cc:cc + args.patch_size].to(device)
                    lq_patch = lq_full[:, :, rr:rr + args.patch_size, cc:cc + args.patch_size].to(device)

                    sr_patch = sample_rddm(trainer, lq_patch, use_amp=args.amp)

                    name = make_patch_name(
                        dataset,
                        image_index,
                        patch_index,
                        keep_original_name=args.keep_original_name,
                    )

                    save_teacher_sample(
                        hr_patch,
                        lq_patch,
                        sr_patch,
                        out_dirs,
                        name,
                        save_residual_png=args.save_residual_png,
                    )
                    save_lr_patch(
                        lr_paths,
                        image_index,
                        rr,
                        cc,
                        args.patch_size,
                        args.scale,
                        out_dirs["lr"] / name,
                    )

                    exported += 1
                    print(
                        f"[{exported}] saved patch teacher output: {name} "
                        f"from image_index={image_index}, top={rr}, left={cc}"
                    )

    return exported


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint_dir) / f"model-{args.milestone}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please check --checkpoint_dir and --milestone."
        )

    out_dirs = ensure_dirs(args.out_dir)

    # Dataset returns [HR, LQ_up].
    # sample=True prevents random crop inside Dataset; patch export crops explicitly below.
    dataset = Dataset(
        [args.hr_flist, args.lq_up_flist],
        image_size=args.image_size,
        augment_flip=False,
        convert_image_to="RGB",
        condition=1,
        equalizeHist=False,
        crop_patch=True,
        sample=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    lr_paths = read_flist(args.lr_flist)

    diffusion, condition = build_diffusion(args)

    # Reuse Trainer.load to correctly restore both model and EMA states.
    # The folder argument only needs to be valid enough for Trainer initialization.
    trainer = Trainer(
        diffusion,
        [args.hr_flist, args.lq_up_flist, args.hr_flist, args.lq_up_flist],
        train_batch_size=1,
        num_samples=1,
        train_lr=8e-5,
        train_num_steps=1,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=args.amp,
        convert_image_to="RGB",
        condition=condition,
        save_and_sample_every=1,
        equalizeHist=False,
        crop_patch=True,
        generation=False,
        results_folder=args.checkpoint_dir,
    )

    trainer.load(args.milestone)
    trainer.ema.ema_model.eval()

    device = trainer.device

    if args.export_mode == "full":
        exported = export_full(args, trainer, loader, dataset, lr_paths, out_dirs, device)
    elif args.export_mode == "patch":
        exported = export_patch(args, trainer, loader, dataset, lr_paths, out_dirs, device)
    else:
        raise ValueError(f"Unsupported export_mode: {args.export_mode}")

    print(f"Done. Exported {exported} samples to: {args.out_dir}")


if __name__ == "__main__":
    main()
