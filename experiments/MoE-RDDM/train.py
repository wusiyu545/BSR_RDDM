import os
# 必须放在任何 torch / src.* 导入之前
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import glob
import re
import torch

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (
    ResidualDiffusion,
    Trainer,
    Unet,
    UnetRes,
    set_seed
)

AUTO_RESUME = False          # 是否自动恢复最新 checkpoint
RESUME_MILESTONE = None      # 手动指定恢复的权重，如 2 表示 model-2.pt
TEST_MILESTONE = None        # 手动指定测试权重，如 2 表示 model-2.pt

# =========================================================
# 基础初始化
# =========================================================
sys.stdout.flush()
set_seed(10)

debug = False

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current_device =", torch.cuda.current_device())
    print("device_name =", torch.cuda.get_device_name(torch.cuda.current_device()))

if debug:
    save_and_sample_every = 2
    sampling_timesteps = 50
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 200
else:
    save_and_sample_every = 5000
    if len(sys.argv) > 1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 50
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 20000   # 总步数

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True
    input_condition = True   # 激活 SDT / expert target 分支
    input_condition_mask = False

# ==========================================
# 训练超参数
# ==========================================
gradient_accumulate_every = 4

# =========================================================
# 数据路径配置
# =========================================================
# 目标：把训练 GT 从 DIV2K_x4_min 改成 DF2K HR
#
# 你需要准备：
#   ./data/DF2K_HR/train_hr.flist
#
# 其中 train_hr.flist 每一行都是一张 DF2K HR 图像路径
# （DIV2K_HR + Flickr2K_HR 合并后的训练集）
#
# 说明：
# 1) 在你刚替换的 base.py 里，训练阶段会“在线退化”，因此 folder[1]
#    这里只是为了满足长度一致检查，直接与 folder[0] 指向同一个 flist 即可。
# 2) 测试部分这里先沿用你当前已有的 test_gt.flist / test_input.flist，
#    这样不会一下子把测试链路改掉。
# =========================================================
if condition:
    train_hr_flist = "./data/DF2K/train_hr.flist"
    val_hr_flist = "./data/benchmark_flist/DF2KVal_hr.flist"

    folder = [
        train_hr_flist,  # train GT / HR
        train_hr_flist,  # 占位，训练时在线退化
        val_hr_flist,  # val GT / HR
        val_hr_flist  # 占位，验证时在线 bicubic x4
    ]

    train_batch_size = 2
    num_samples = 1
    sum_scale = 0.01
    image_size = 256

else:
    folder = "/home/liu/disk12t/liu_data/dataset/CelebA/img_align_celeba"
    train_batch_size = 32
    num_samples = 25
    sum_scale = 1
    image_size = 32


def count_lines(path):
    if not os.path.exists(path):
        return -1
    with open(path, "r", encoding="utf-8") as f:
        return len([line for line in f.readlines() if line.strip()])


def check_required_paths(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("\n❌ [路径检查失败] 以下文件/路径不存在：")
        for p in missing:
            print("   -", p)
        raise FileNotFoundError("请先准备好上述数据路径后再运行。")


# =========================================================
# 模型构建
# =========================================================
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
        input_condition=input_condition
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
        input_condition_mask=input_condition_mask
    )


if __name__ == "__main__":
    if condition:
        print("\n================ 数据检查 ================")
        check_required_paths([folder[0], folder[2], folder[3]])

        print(f"[Train HR flist] {folder[0]}")
        train_count = count_lines(folder[0])
        print(f"训练 HR 图片数量: {train_count}")

        print(f"[Test GT flist]  {folder[2]}")
        test_gt_count = count_lines(folder[2])
        print(f"测试 GT 图片数量: {test_gt_count}")

        print(f"[Test IN flist]  {folder[3]}")
        test_in_count = count_lines(folder[3])
        print(f"测试 Input 图片数量: {test_in_count}")

        if train_count <= 0:
            raise RuntimeError("train_hr.flist 为空，无法开始训练。")

        if test_gt_count != test_in_count:
            print("⚠️ [警告] test_gt 与 test_input 数量不一致，请确认测试集配对。")

    results_dir = "./results/MoE_SDT_BlindSR_x4_DF2K_alignMCD"

    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=train_batch_size,
        num_samples=num_samples,
        train_lr=8e-5,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=gradient_accumulate_every,
        ema_decay=0.995,
        amp=True,
        fp16=True,
        convert_image_to="RGB",
        condition=condition,
        save_and_sample_every=save_and_sample_every,
        equalizeHist=False,
        crop_patch=True,
        generation=False,
        results_folder=results_dir
    )

    if trainer.accelerator.is_local_main_process:
        if RESUME_MILESTONE is not None:
            print(f"\n🎯 [手动加载] 正在加载指定权重 model-{RESUME_MILESTONE}.pt ...")
            trainer.load(RESUME_MILESTONE)

        elif AUTO_RESUME:
            model_files = glob.glob(f"{results_dir}/model-*.pt")
            if len(model_files) > 0:
                milestones = [
                    int(re.findall(r"model-(\d+)\.pt", f)[0])
                    for f in model_files
                    if re.findall(r"model-(\d+)\.pt", f)
                ]
                best_milestone = max(milestones) if milestones else 0
                print(f"\n🔄 找到最新 Checkpoint: model-{best_milestone}.pt，正在加载以恢复训练...")
                trainer.load(best_milestone)
            else:
                print("\n🚀 [信息] 没有找到 checkpoint，从零开始训练。")
        else:
            print("\n🚀 [信息] 已关闭自动恢复，不加载任何旧权重，从零开始训练。")

    # =========================================================
    # 开始训练
    # =========================================================
    trainer.train()

    # =========================================================
    # 训练结束后自动加载最新权重测试
    # =========================================================
    if trainer.accelerator.is_local_main_process:
        if TEST_MILESTONE is not None:
            print(f"\n✅ [测试准备] 正在加载指定测试权重 model-{TEST_MILESTONE}.pt ...")
            trainer.load(TEST_MILESTONE)
        else:
            model_files = glob.glob(f"{results_dir}/model-*.pt")
            if len(model_files) > 0:
                milestones = [
                    int(re.findall(r"model-(\d+)\.pt", f)[0])
                    for f in model_files
                    if re.findall(r"model-(\d+)\.pt", f)
                ]
                best_milestone = max(milestones) if milestones else 0
                print(f"\n✅ [测试准备] 正在加载最终权重 model-{best_milestone}.pt 进行测试...")
                trainer.load(best_milestone)

        trainer.set_results_folder(
            "./results/MoE_SDT_BlindSR_x4_DF2K_alignMCD_test_timestep_" + str(sampling_timesteps)
        )
        trainer.test(last=True)