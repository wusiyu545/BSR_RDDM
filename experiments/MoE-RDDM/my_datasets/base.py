import os
import random
from pathlib import Path

import Augmentor
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def _apply_jpeg(img_uint8, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, encimg = cv2.imencode(".jpg", img_uint8, encode_param)
    if not success:
        return img_uint8
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def _random_motion_kernel(kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
    center = (kernel_size - 1) / 2.0

    angle = random.uniform(0, np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x = xs - center
    y = ys - center
    dist = np.abs(-sin_a * x + cos_a * y)

    kernel[dist < 0.5] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s
    else:
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
    return kernel


def _random_gaussian_kernel():
    """
    返回:
        kernel: 卷积核
        blur_type: 'iso' / 'aniso'
        blur_strength: 0~1 粗略强度
    """
    ksize = random.choice([7, 9, 11, 13, 15, 17, 19, 21])

    if random.random() < 0.5:
        sigma = random.uniform(0.2, 3.0)
        k1d = cv2.getGaussianKernel(ksize, sigma)
        kernel = np.outer(k1d, k1d).astype(np.float32)
        blur_strength = min(sigma / 3.0, 1.0)
        return kernel, "iso", blur_strength

    sigma_x = random.uniform(0.5, 4.0)
    sigma_y = random.uniform(0.5, 4.0)
    theta = random.uniform(0, np.pi)

    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)

    kernel = np.exp(-0.5 * ((xr ** 2) / (sigma_x ** 2) + (yr ** 2) / (sigma_y ** 2)))
    kernel = kernel.astype(np.float32)
    kernel /= np.sum(kernel)

    blur_strength = min(max(sigma_x, sigma_y) / 4.0, 1.0)
    return kernel, "aniso", blur_strength


def _sample_blur_kernel():
    """
    统一 blur 采样入口，便于 first-order / second-order 复用
    """
    if random.random() < 0.75:
        return _random_gaussian_kernel()

    kernel = _random_motion_kernel(kernel_size=random.choice([9, 11, 13, 15]))
    blur_type = "motion"
    blur_strength = 0.8
    return kernel, blur_type, blur_strength


def build_expert_target_sr_x4(deg_info):
    """
    4 专家 soft target
    E0: compression / ringing
    E1: isotropic / mild blur
    E2: directional / strong blur
    E3: downsample-driven detail recovery + stronger noise
    """
    t = np.zeros(4, dtype=np.float32)

    jpeg_strength = deg_info.get("jpeg_strength", 0.0)
    blur_type = deg_info.get("blur_type", "none")
    blur_strength = deg_info.get("blur_strength", 0.0)
    resize_strength = deg_info.get("resize_strength", 0.0)
    noise_strength = deg_info.get("noise_strength", 0.0)

    # E0: compression / ringing
    t[0] += 1.2 * jpeg_strength

    # E1 / E2: blur split
    if blur_type == "iso":
        # isotropic blur -> mainly E1
        t[1] += 1.0 * blur_strength

    elif blur_type == "aniso":
        # weak anisotropic -> E1
        # strong anisotropic -> E2
        if blur_strength < 0.55:
            t[1] += 0.9 * blur_strength
            t[2] += 0.2 * blur_strength
        else:
            t[1] += 0.3 * blur_strength
            t[2] += 1.0 * blur_strength

    elif blur_type == "motion":
        # motion blur -> mainly E2
        t[2] += 1.1 * blur_strength

    # E3: downsample + stronger noise + detail regeneration
    # 不要让 resize 一票压死所有样本
    t[3] += 0.55 * resize_strength
    t[3] += 0.75 * noise_strength

    # 如果压缩很弱且 blur 很弱，但有下采样，E3 保底承担细节补偿
    if resize_strength > 0 and blur_strength < 0.25 and jpeg_strength < 0.2:
        t[3] += 0.15

    t += 0.05
    t /= t.sum()
    return t.astype(np.float32)


class Dataset(TorchDataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=("jpg", "jpeg", "png", "tiff"),
        augment_flip=False,
        convert_image_to=None,
        condition=0,
        equalizeHist=False,
        crop_patch=True,
        sample=False,
    ):
        super().__init__()

        # blind SR x4 配置
        self.sr_scale = 4
        self.two_order_deg = True

        self.equalizeHist = equalizeHist
        self.exts = tuple([e.lower() for e in exts])
        self.augment_flip = augment_flip
        self.condition = condition
        self.crop_patch = crop_patch
        self.sample = sample
        self.image_size = image_size
        self.convert_image_to = convert_image_to

        if condition == 1:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            assert len(self.gt) == len(self.input), (
                f"❌ [数据致命错误] GT 数量({len(self.gt)}) 与 Input 数量({len(self.input)}) 不一致！"
            )

        elif condition == 0:
            self.paths = self.load_flist(folder)

        elif condition == 2:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self.input_condition = self.load_flist(folder[2])
            assert len(self.gt) == len(self.input) == len(self.input_condition), (
                f"❌ [数据致命错误] 三路输入数据集长度不一致: "
                f"GT({len(self.gt)}), Input({len(self.input)}), Cond({len(self.input_condition)})"
            )

    def __len__(self):
        if self.condition in [1, 2]:
            return len(self.input)
        return len(self.paths)

    def degrade_for_blind_sr_x4(self, img_gt):
        """
        img_gt: HWC, uint8, BGR, 范围 0~255
        return:
            img_lq_up: 与 GT 同尺寸的退化输入
            expert_target: 4 维 soft label
        """
        img = img_gt.astype(np.float32)
        h, w = img.shape[:2]

        deg_info = {
            "blur_type": "none",
            "blur_strength": 0.0,
            "resize_strength": 0.0,
            "noise_strength": 0.0,
            "jpeg_strength": 0.0,
        }

        # 1) first-order blur
        if random.random() < 0.9:
            kernel, blur_type, blur_strength = _sample_blur_kernel()
            img = cv2.filter2D(img, -1, kernel)
            deg_info["blur_type"] = blur_type
            deg_info["blur_strength"] = blur_strength

        # 2) mild pre-resize perturbation
        # 为了更贴近 x4 blind SR，范围收紧一点
        if random.random() < 0.5:
            resize_factor = random.uniform(0.95, 1.05)
            inter_mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
            tmp_w = max(1, int(round(w * resize_factor)))
            tmp_h = max(1, int(round(h * resize_factor)))
            img = cv2.resize(img, (tmp_w, tmp_h), interpolation=inter_mode)

        # 3) force native x4 LR
        lr_w = max(1, w // self.sr_scale)
        lr_h = max(1, h // self.sr_scale)
        inter_mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
        img = cv2.resize(img, (lr_w, lr_h), interpolation=inter_mode)
        deg_info["resize_strength"] = 1.0

        # 4) noise on LR
        if random.random() < 0.8:
            noise_level = random.uniform(1.0, 12.0)
            noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
            img = img + noise
            deg_info["noise_strength"] = (noise_level - 1.0) / 11.0

        # 5) JPEG on LR
        img = np.clip(img, 0, 255).astype(np.uint8)
        if random.random() < 0.8:
            q = random.randint(30, 95)
            img = _apply_jpeg(img, q)
            deg_info["jpeg_strength"] = (95.0 - q) / 65.0

        # 6) optional second-order light degradation
        if self.two_order_deg and random.random() < 0.5:
            # second blur
            if random.random() < 0.5:
                kernel2, blur_type2, blur_strength2 = _sample_blur_kernel()
                img = cv2.filter2D(img, -1, kernel2)

                # 用“更强”的 blur 更新标签，避免第二次 blur 完全丢失在 expert_target 里
                if blur_strength2 >= deg_info["blur_strength"]:
                    deg_info["blur_type"] = blur_type2
                    deg_info["blur_strength"] = blur_strength2

            # second noise
            if random.random() < 0.5:
                noise_level2 = random.uniform(0.5, 5.0)
                noise2 = np.random.normal(0, noise_level2, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise2, 0, 255).astype(np.uint8)
                deg_info["noise_strength"] = max(
                    deg_info["noise_strength"],
                    min((noise_level2 - 0.5) / 4.5, 1.0),
                )

            # second JPEG
            if random.random() < 0.5:
                q2 = random.randint(50, 95)
                img = _apply_jpeg(img, q2)
                deg_info["jpeg_strength"] = max(
                    deg_info["jpeg_strength"],
                    (95.0 - q2) / 45.0,
                )

        # 7) upsample back to GT size
        up_mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        img_lq_up = cv2.resize(img, (w, h), interpolation=up_mode)
        img_lq_up = np.ascontiguousarray(img_lq_up)

        expert_target = build_expert_target_sr_x4(deg_info)
        return img_lq_up, expert_target

    def __getitem__(self, index):
        if self.condition == 1:
            img0_pil = Image.open(self.gt[index])
            img0_pil = convert_image_to_fn(self.convert_image_to, img0_pil) if self.convert_image_to else img0_pil

            if self.sample:
                img1_pil = Image.open(self.input[index])
                img1_pil = convert_image_to_fn(self.convert_image_to, img1_pil) if self.convert_image_to else img1_pil
                expert_target = None
            else:
                img_gt = cv2.cvtColor(np.array(img0_pil), cv2.COLOR_RGB2BGR)
                img_lq, expert_target = self.degrade_for_blind_sr_x4(img_gt)
                img1_pil = Image.fromarray(cv2.cvtColor(img_lq.astype(np.uint8), cv2.COLOR_BGR2RGB))

            img0, img1 = self.pad_img([img0_pil, img1_pil], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1 = self.get_patch([img0, img1], self.image_size)

            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            images = [[img0, img1]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)

            if not self.crop_patch:
                h, w = img0.shape[:2]
                if h != self.image_size or w != self.image_size:
                    p.resize(1, self.image_size, self.image_size)

            g = p.generator(batch_size=1)
            augmented_images = next(g)

            img0_out = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            img1_out = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)

            if self.sample:
                return [self.to_tensor(img0_out), self.to_tensor(img1_out)]

            return [
                self.to_tensor(img0_out),
                self.to_tensor(img1_out),
                torch.tensor(expert_target, dtype=torch.float32),
            ]

        elif self.condition == 0:
            path = self.paths[index]
            img = Image.open(path)
            img = convert_image_to_fn(self.convert_image_to, img) if self.convert_image_to else img

            img = self.pad_img([img], self.image_size)[0]

            if self.crop_patch and not self.sample:
                img = self.get_patch([img], self.image_size)[0]

            img = self.cv2equalizeHist(img) if self.equalizeHist else img

            images = [[img]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)

            if not self.crop_patch:
                h, w = img.shape[:2]
                if h != self.image_size or w != self.image_size:
                    p.resize(1, self.image_size, self.image_size)

            g = p.generator(batch_size=1)
            augmented_images = next(g)
            img = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)

            return self.to_tensor(img)

        elif self.condition == 2:
            img0 = Image.open(self.gt[index])
            img1 = Image.open(self.input[index])
            img2 = Image.open(self.input_condition[index])

            img0 = convert_image_to_fn(self.convert_image_to, img0) if self.convert_image_to else img0
            img1 = convert_image_to_fn(self.convert_image_to, img1) if self.convert_image_to else img1
            img2 = convert_image_to_fn(self.convert_image_to, img2) if self.convert_image_to else img2

            img0, img1, img2 = self.pad_img([img0, img1, img2], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1, img2 = self.get_patch([img0, img1, img2], self.image_size)

            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            images = [[img0, img1, img2]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)

            if not self.crop_patch:
                h, w = img0.shape[:2]
                if h != self.image_size or w != self.image_size:
                    p.resize(1, self.image_size, self.image_size)

            g = p.generator(batch_size=1)
            augmented_images = next(g)

            img0 = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(augmented_images[0][2], cv2.COLOR_BGR2RGB)

            return [self.to_tensor(img0), self.to_tensor(img1), self.to_tensor(img2)]

        raise ValueError(f"Unsupported condition: {self.condition}")

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                files = []
                for ext in self.exts:
                    files.extend([str(p) for p in Path(flist).glob(f"**/*.{ext}")])
                return sorted(files)

            if os.path.isfile(flist):
                try:
                    with open(flist, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f.readlines()]
                        lines = [l for l in lines if len(l) > 0]
                        return lines
                except Exception as e:
                    print(f"[Warning] Cannot read {flist} as text file. Treating as image path. Error: {e}")
                    return [flist]

        return []

    def cv2equalizeHist(self, img):
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir + "_" + os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h - patch_size)
        cc = random.randint(0, w - patch_size)

        out = []
        for img in image_list:
            out.append(img[rr:rr + patch_size, cc:cc + patch_size, :])
        return out

    def pad_img(self, img_list, patch_size, block_size=8):
        out = []
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]

            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size - h
                h = patch_size
            if w < patch_size:
                right = patch_size - w
                w = patch_size

            bottom = bottom + (h // block_size) * block_size + (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + (block_size if w % block_size != 0 else 0) - w

            out.append(
                cv2.copyMakeBorder(
                    img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            )
        return out

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size - h
            h = patch_size
        if w < patch_size:
            right = patch_size - w
            w = patch_size

        bottom = bottom + (h // block_size) * block_size + (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]