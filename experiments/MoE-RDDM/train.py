import os
# 必须放在任何 torch / src.* 导入之前
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import glob
import re
import torch
from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)

AUTO_RESUME = False          # 是否自动恢复最新checkpoint
RESUME_MILESTONE = False      # 手动指定恢复的权重，如 2 表示 model-2.pt
AUTO_TEST_LOAD_LATEST = False
TEST_MILESTONE = None        # 手动指定测试权重，如 2 表示 model-2.pt

# init

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
    save_and_sample_every = 1000
    if len(sys.argv) > 1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 50
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 20000#总步数

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True
    input_condition = True  # 激活 SDT 分支
    input_condition_mask = False

# ==========================================
# 🎯 提取训练超参数到外层，方便两阶段缩放
# ==========================================
gradient_accumulate_every = 8  # 提取到这里统一管理

if condition:
    base_path = "./data/FFHQ_512"
    folder = [
        f"{base_path}/train_gt.flist",
        f"{base_path}/train_gt.flist",
        f"{base_path}/test_gt.flist",
        f"{base_path}/test_input.flist"
    ]
    train_batch_size = 1
    num_samples = 4
    sum_scale = 0.01
    image_size = 512
else:
    folder = '/home/liu/disk12t/liu_data/dataset/CelebA/img_align_celeba'
    train_batch_size = 32
    num_samples = 25
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
    # 打印测试集检查
    print(f"DEBUG: 正在检查验证列表：{folder[2]}")
    try:
        with open(folder[2], 'r', encoding='utf-8') as f:
            print(f"DEBUG: 验证集图片数量：{len([line for line in f.readlines() if line.strip()])}")
    except FileNotFoundError:
        print("⚠️ 警告：找不到 test.flist 文件！")

    # =========================================================
    # 🚀 BFR 全自动控制台：智能感知进度，无缝切换 Stage
    # =========================================================
    # ⚠️ 这里的缩进已经修复，确保它在所有情况下都能正常执行
    import glob
    import re

    STAGE2_START_STEP = 999999  # 设定自动开启 Stage 2 的触发步数
    current_best_step = 0
    results_dir = './results/MoE_SDT_Stage1_Debug'

    # 1. 发射雷达：扫描 checkpoints，探测当前进度
    if os.path.exists(results_dir):
        model_files = glob.glob(f"{results_dir}/model-*.pt")
        if len(model_files) > 0:
            milestones = [int(re.findall(r'model-(\d+)\.pt', f)[0]) for f in model_files if
                          re.findall(r'model-(\d+)\.pt', f)]
            best_milestone = max(milestones) if milestones else 0
            current_best_step = best_milestone * save_and_sample_every

    # 2. 自动裁决：该用哪个 Stage？
    training_stage = 2 if current_best_step >= STAGE2_START_STEP else 1

    # 3. 自适应硬件与网络加载
    # 3. 自适应硬件与网络加载
    if training_stage == 2:
        print(f"\n🔥 [自动感知] 当前进度 ({current_best_step} 步) 已达到 Stage 2 要求！")
        print("🔥 [高阶开启] 自动启动人脸保真微调：显式加载 LPIPS & FaceID 约束！")

        # 🚀 极致懒加载：只有真正需要跑 Stage 2 时，才向环境中索要这两个大模型包
        from src.bfr_loss import BFR_Stage2_Loss

        diffusion.use_stage2_loss = True
        diffusion.bfr_loss_fn = BFR_Stage2_Loss(device='cuda')

        # 自动调整硬件负载，规避显存爆炸
        train_batch_size = max(1, train_batch_size // 2)
        gradient_accumulate_every = gradient_accumulate_every * 2
        print(f"   -> 硬件保护触发：Batch Size 自动缩放至 {train_batch_size}, 梯度累加至 {gradient_accumulate_every}")
    else:
        print(f"\n🌱 [自动感知] 当前进度 ({current_best_step} 步)，未达到 Stage 2 阈值 ({STAGE2_START_STEP})。")
        print("🌱 [基础构建] 自动执行 Stage 1：基础去噪与专家路由分化训练。")

    # =========================================================

    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=train_batch_size,
        num_samples=num_samples,
        train_lr=8e-5,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=gradient_accumulate_every,  # 使用变量
        ema_decay=0.995,
        amp=True,
        fp16=True,
        convert_image_to="RGB",
        condition=condition,
        save_and_sample_every=save_and_sample_every,
        equalizeHist=False,
        crop_patch=False,
        generation=False,
        results_folder=results_dir # 统一使用 results_dir
    )

    if trainer.accelerator.is_local_main_process:
        if RESUME_MILESTONE is not None:
            print(f"\n🎯 [手动加载] 正在加载指定权重 model-{RESUME_MILESTONE}.pt ...")
            trainer.load(RESUME_MILESTONE)

        elif AUTO_RESUME:
            model_files = glob.glob(f"{results_dir}/model-*.pt")
            if len(model_files) > 0:
                milestones = [int(re.findall(r'model-(\d+)\.pt', f)[0]) for f in model_files if
                              re.findall(r'model-(\d+)\.pt', f)]
                best_milestone = max(milestones) if milestones else 0
                print(f"\n🔄 找到最新 Checkpoint: model-{best_milestone}.pt，正在加载以恢复训练...")
                trainer.load(best_milestone)
            else:
                print("\n🚀 [信息] 没有找到checkpoint，从零开始训练。")

        else:
            print("\n🚀 [信息] 已关闭自动恢复，不加载任何旧权重，从零开始训练。")

    # 开始大考
    trainer.train()

    # =========================================================
    # 🚀 智能加载大结局测试
    # =========================================================
    if trainer.accelerator.is_local_main_process:
        model_files = glob.glob(f"{results_dir}/model-*.pt")
        if len(model_files) > 0:
            milestones = [int(re.findall(r'model-(\d+)\.pt', f)[0]) for f in model_files if
                          re.findall(r'model-(\d+)\.pt', f)]
            best_milestone = max(milestones) if milestones else 0
            print(f"\n✅ [测试准备] 正在加载最终权重 model-{best_milestone}.pt 进行跑分测试...")
            trainer.load(best_milestone)

        trainer.set_results_folder('./results/test_timestep_' + str(sampling_timesteps))
        trainer.test(last=True)