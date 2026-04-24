import os
import sys
import glob
import re

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (
    ResidualDiffusion,
    Trainer,
    Unet,
    UnetRes,
    set_seed,
)

# init
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
sys.stdout.flush()
set_seed(10)

debug = False
if debug:
    save_and_sample_every = 10
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 20  # 占位，test 不训练
else:
    save_and_sample_every = 10
    if len(sys.argv) > 1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 50
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 20  # 占位，test 不训练

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True
    input_condition = True
    input_condition_mask = False

if condition:

    #测试集
    BENCHMARK_NAME = "Set5"

    train_hr_flist = "./data/DF2K/train_hr.flist"
    test_hr_flist = f"./data/benchmark_flist/{BENCHMARK_NAME}_hr.flist"

    folder = [
        train_hr_flist,  # 占位
        train_hr_flist,  # 占位
        test_hr_flist,  # test GT / HR
        test_hr_flist  # 占位，测试时在线 bicubic x4
    ]




    train_batch_size = 1   # 这里只是 Trainer 初始化占位
    num_samples = 1        # 测试逐张输出，最省显存
    sum_scale = 0.01
    image_size = 256
else:
    folder = '/home/liu/disk12t/liu_data/dataset/CelebA/img_align_celeba'
    train_batch_size = 32
    num_samples = 1
    sum_scale = 1
    image_size = 32

if original_ddim_ddpm:
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps_original_ddim_ddpm,
        loss_type='l1',
    )
else:
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        share_encoder=1,
        condition=condition,
        input_condition=input_condition,
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        objective='pred_res_noise',
        loss_type='l1',
        condition=condition,
        sum_scale=sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
    )

# 必须和当前 train.py 的 results_dir 一致
results_dir = './results/MoE_SDT_Stage1_Debug'

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=8e-5,
    train_num_steps=train_num_steps,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    amp=False,
    fp16=False,
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=False,
    results_folder=results_dir,
)

if __name__ == "__main__":
    # 简单检查 flist 是否存在
    print(f"DEBUG: 正在检查测试 GT 列表：{folder[2]}")
    try:
        with open(folder[2], 'r', encoding='utf-8') as f:
            print(f"DEBUG: 测试 GT 数量：{len([line for line in f.readlines() if line.strip()])}")
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到测试 GT 列表: {folder[2]}")

    if trainer.accelerator.is_local_main_process:
        model_files = glob.glob(f"{results_dir}/model-*.pt")

        if len(model_files) > 0:
            milestones = [
                int(re.findall(r'model-(\d+)\.pt', f)[0])
                for f in model_files if re.findall(r'model-(\d+)\.pt', f)
            ]
            best_milestone = max(milestones) if milestones else 0
            print(f"\n🚀 [智能加载] 找到最新权重: model-{best_milestone}.pt，正在加载...")
            trainer.load(best_milestone)
        else:
            raise FileNotFoundError(f"\n❌ 没有在 {results_dir} 里找到任何 model-*.pt")

        # 单独输出到测试目录，避免覆盖训练目录
        trainer.set_results_folder(f'./results/test_stage1_debug_t{sampling_timesteps}')
        trainer.test(last=True)
