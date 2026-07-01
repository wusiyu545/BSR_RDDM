import argparse
import os
import sys

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.get_dataset import dataset as build_dataset
from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (
    ResidualDiffusion,
    Trainer,
    Unet,
    UnetRes,
    cycle,
    set_seed,
)


class BSRTrainer(Trainer):
    """Trainer wrapper for RDDM-BSR teacher.

    Compared with the original Trainer, this wrapper makes two BSR-specific
    changes without touching the RDDM model itself:

    1. Periodic validation samples are cropped to image_size patches. This
       avoids running RDDM sampling on full DIV2K images, which can easily cause
       CUDA/CUBLAS failures or OOM.
    2. A checkpoint is saved at every save_and_sample_every milestone. This is
       more convenient for debug runs, e.g. 200 steps with save_every=50 will
       produce model-1.pt ... model-4.pt.
    """

    def use_patch_sample_loader(self, folder, args):
        if not self.condition or len(folder) != 4:
            return

        val_dataset = build_dataset(
            folder[2:4],
            self.image_size,
            augment_flip=False,
            convert_image_to=args.convert_image_to,
            condition=1,
            equalizeHist=args.equalize_hist,
            crop_patch=True,
            sample=False,
            generation=False,
        )

        self.sample_dataset = val_dataset
        self.sample_loader = cycle(
            self.accelerator.prepare(
                DataLoader(
                    self.sample_dataset,
                    batch_size=self.num_samples,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=args.sample_num_workers,
                )
            )
        )

        if self.accelerator.is_main_process:
            print(
                "[BSRTrainer] Periodic validation sampling uses "
                f"{self.image_size}x{self.image_size} patches."
            )

    def train(self):
        accelerator = self.accelerator

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss = total_loss + loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        self.sample(milestone)

                pbar.set_description(f"loss: {total_loss:.4f}")
                pbar.update(1)

        accelerator.print("training complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RDDM as a Blind Super-Resolution teacher."
    )

    # Data paths: paired restoration format
    # imgs[0] = HR, imgs[1] = LQ_up
    parser.add_argument("--train_hr", type=str, required=True,
                        help="Path to training HR flist.")
    parser.add_argument("--train_lq_up", type=str, required=True,
                        help="Path to training LQ-upsampled flist.")
    parser.add_argument("--val_hr", type=str, required=True,
                        help="Path to validation HR flist.")
    parser.add_argument("--val_lq_up", type=str, required=True,
                        help="Path to validation LQ-upsampled flist.")

    # Basic training settings
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES setting, e.g. '0'.")
    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Training HR patch size. For x4 SR, LR patch is usually 64.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to save during periodic sampling. Must be a square number.")
    parser.add_argument("--train_steps", type=int, default=120000,
                        help="Total training steps.")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save/sample interval in steps.")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=8e-5,
                        help="Learning rate.")

    # RDDM settings
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion training timesteps.")
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of DDIM sampling timesteps.")
    parser.add_argument("--sum_scale", type=float, default=1.0,
                        help="RDDM residual/noise scale for conditional restoration.")
    parser.add_argument("--objective", type=str, default="pred_res_noise",
                        choices=[
                            "pred_res_noise",
                            "pred_res_add_noise",
                            "pred_x0_noise",
                            "pred_x0_add_noise",
                            "pred_noise",
                            "pred_res",
                        ],
                        help="RDDM training objective.")
    parser.add_argument("--loss_type", type=str, default="l1",
                        choices=["l1", "l2"],
                        help="Loss type for RDDM.")

    # Model settings
    parser.add_argument("--dim", type=int, default=64,
                        help="Base channel dimension of U-Net.")
    parser.add_argument("--share_encoder", type=int, default=0,
                        choices=[-1, 0, 1],
                        help="UnetRes share_encoder setting. Keep 0 to match original restoration template.")

    # Checkpoint / output
    parser.add_argument("--save_dir", type=str, default="./results/rddm_bsr_teacher",
                        help="Directory to save checkpoints and samples.")
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume milestone. 0 means train from scratch. For example, 80 loads model-80.pt.")
    parser.add_argument("--test_after_train", action="store_true",
                        help="Run test after training is completed.")

    # Precision and data processing
    parser.add_argument("--amp", action="store_true",
                        help="Enable native AMP in Trainer.")
    parser.set_defaults(crop_patch=True)
    parser.add_argument("--crop_patch", dest="crop_patch", action="store_true",
                        help="Enable random crop for paired HR/LQ_up patches during training. Default: enabled.")
    parser.add_argument("--no_crop_patch", dest="crop_patch", action="store_false",
                        help="Disable random crop. Use this only for full-image training or testing.")
    parser.add_argument("--sample_num_workers", type=int, default=0,
                        help="DataLoader workers for periodic patch sampling. 0 is safer on Windows.")
    parser.add_argument("--equalize_hist", action="store_true",
                        help="Use histogram equalization for input images. Normally keep False for BSR.")
    parser.add_argument("--convert_image_to", type=str, default="RGB",
                        help="Image mode conversion, usually RGB.")

    # Optional original DDPM path, kept for compatibility but not recommended for BSR teacher.
    parser.add_argument("--original_ddim_ddpm", action="store_true",
                        help="Use original GaussianDiffusion instead of ResidualDiffusion. Not recommended for BSR teacher.")

    return parser.parse_args()


def build_diffusion(args):
    condition = not args.original_ddim_ddpm
    input_condition = False
    input_condition_mask = False

    if args.original_ddim_ddpm:
        model = Unet(
            dim=args.dim,
            dim_mults=(1, 2, 4, 8),
        )
        diffusion = GaussianDiffusion(
            model,
            image_size=args.image_size,
            timesteps=args.timesteps,
            sampling_timesteps=args.sampling_timesteps,
            loss_type=args.loss_type,
        )
    else:
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


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.stdout.flush()
    set_seed(args.seed)

    # Paired restoration format used by the existing base.Dataset:
    # folder[0] = train HR
    # folder[1] = train LQ_up
    # folder[2] = val HR
    # folder[3] = val LQ_up
    #
    # This order is critical because RDDM internally uses:
    #   x_start = imgs[0] = HR
    #   x_input = imgs[1] = LQ_up
    #   x_res   = x_input - x_start = LQ_up - HR
    folder = [
        args.train_hr,
        args.train_lq_up,
        args.val_hr,
        args.val_lq_up,
    ]

    diffusion, condition = build_diffusion(args)

    trainer = BSRTrainer(
        diffusion,
        folder,
        train_batch_size=args.batch_size,
        num_samples=args.num_samples,
        train_lr=args.lr,
        train_num_steps=args.train_steps,
        gradient_accumulate_every=args.grad_accum,
        ema_decay=0.995,
        amp=args.amp,
        convert_image_to=args.convert_image_to,
        condition=condition,
        save_and_sample_every=args.save_every,
        equalizeHist=args.equalize_hist,
        crop_patch=args.crop_patch,
        generation=False,
        results_folder=args.save_dir,
    )

    # The original Trainer uses full validation images for periodic sampling.
    # For DIV2K / DF2K BSR, this is too memory-intensive for RDDM sampling.
    # Replace it with patch sampling before training starts.
    trainer.use_patch_sample_loader(folder, args)

    if trainer.accelerator.is_local_main_process and args.resume > 0:
        trainer.load(args.resume)

        # Force reset learning rate after loading optimizer state.
        for param_group in trainer.opt.param_groups:
            param_group["lr"] = args.lr

        print(f"[resume] reset optimizer lr to {args.lr}")

        # Force reset learning rate after loading optimizer state.
        for param_group in trainer.opt.param_groups:
            param_group["lr"] = args.lr

        print(f"[resume] reset optimizer lr to {args.lr}")
    
    trainer.train()

    if args.test_after_train:
        if trainer.accelerator.is_local_main_process:
            final_milestone = trainer.train_num_steps // trainer.save_and_sample_every
            trainer.load(final_milestone)
            trainer.set_results_folder(
                os.path.join(args.save_dir, "test_timestep_{}".format(args.sampling_timesteps))
            )
            trainer.test(last=True)


if __name__ == "__main__":
    main()
