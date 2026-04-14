import os
import random
import math
from pathlib import Path

import Augmentor
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# ==================== SCI 标准模糊核生成器 ====================
def get_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, angle):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    theta = angle * np.pi / 180.
    xx_rot = np.cos(theta) * xx - np.sin(theta) * yy
    yy_rot = np.sin(theta) * xx + np.cos(theta) * yy

    kernel = np.exp(-(xx_rot ** 2 / (2. * sigma_x ** 2) + yy_rot ** 2 / (2. * sigma_y ** 2)))
    return kernel / np.sum(kernel)


def get_motion_blur_kernel(kernel_size, angle):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    return kernel / np.sum(kernel)

def build_expert_target(deg_info):
    """
    E0: JPEG / compression
    E1: anisotropic Gaussian blur
    E2: motion blur
    E3: resize + noise + detail recovery
    """
    t = np.zeros(4, dtype=np.float32)

    # E0: JPEG
    t[0] += 1.2 * deg_info["jpeg_strength"]

    # E1 / E2: blur family
    if deg_info["blur_type"] == "aniso":
        t[1] += 1.0 * deg_info["blur_strength"]
    elif deg_info["blur_type"] == "motion":
        t[2] += 1.0 * deg_info["blur_strength"]

    # E3: resize + noise
    t[3] += 0.7 * deg_info["resize_strength"] + 0.7 * deg_info["noise_strength"]

    # 防止全零
    if t.sum() < 1e-8:
        t[:] = 0.25
    else:
        # 轻微保底，避免 one-hot 太硬
        t = t + 0.05
        t = t / t.sum()

    return t
# =================================================================


class Dataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_flip=False,
            convert_image_to=None,
            condition=0,
            equalizeHist=False,
            crop_patch=True,
            sample=False
    ):
        super().__init__()
        self.equalizeHist = equalizeHist
        self.exts = exts
        self.augment_flip = augment_flip
        self.condition = condition
        self.crop_patch = crop_patch
        self.sample = sample

        if condition == 1:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            # 🚀 强硬防守：长度不对立刻终止程序，大声报错！
            assert len(self.gt) == len(self.input), \
                f"❌ [数据致命错误] GT 数量({len(self.gt)}) 与 Input 数量({len(self.input)}) 不一致！请检查你的数据集和 flist 文件！"

        elif condition == 0:
            self.paths = self.load_flist(folder)

        elif condition == 2:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self.input_condition = self.load_flist(folder[2])
            # 🚀 三重检查
            assert len(self.gt) == len(self.input) == len(self.input_condition), \
                f"❌ [数据致命错误] 三路输入数据集长度不一致: GT({len(self.gt)}), Input({len(self.input)}), Cond({len(self.input_condition)})"

        self.image_size = image_size
        self.convert_image_to = convert_image_to

    def __len__(self):
        # 既然我们在 __init__ 里已经发誓它们长度绝对一致了，这里就可以放心地直接返回 input 的长度了
        if self.condition == 1 or self.condition == 2:
            return len(self.input)
        else:
            return len(self.paths)

    def __getitem__(self, index):
        if self.condition == 1:
            img0_pil = Image.open(self.gt[index])
            img0_pil = convert_image_to_fn(self.convert_image_to, img0_pil) if self.convert_image_to else img0_pil

            if self.sample:
                img1_pil = Image.open(self.input[index])
                img1_pil = convert_image_to_fn(self.convert_image_to, img1_pil) if self.convert_image_to else img1_pil
            else:
                img_gt = cv2.cvtColor(np.array(img0_pil), cv2.COLOR_RGB2BGR)
                img_lq = img_gt.copy().astype(np.float32)
                deg_info = {
                    "blur_type": "none",  # none / aniso / motion
                    "blur_strength": 0.0,
                    "resize_strength": 0.0,
                    "noise_strength": 0.0,
                    "jpeg_strength": 0.0,
                }

                # 1. 空间模糊 (Blur) - 80% 概率
                if random.random() < 0.8:
                    kernel_size = random.choice([13, 17, 21, 27, 33, 39, 47])

                    if random.random() < 0.7:
                        sigma_x = random.uniform(3.0, 10.0)
                        sigma_y = random.uniform(3.0, 10.0)
                        angle = random.uniform(0, 180)
                        kernel = get_anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, angle)

                        deg_info["blur_type"] = "aniso"
                        deg_info["blur_strength"] = ((sigma_x + sigma_y) * 0.5 - 3.0) / (10.0 - 3.0)

                    else:
                        angle = random.uniform(0, 360)
                        kernel = get_motion_blur_kernel(kernel_size, angle)

                        deg_info["blur_type"] = "motion"
                        deg_info["blur_strength"] = (kernel_size - 13.0) / (47.0 - 13.0)

                    img_lq = cv2.filter2D(img_lq, -1, kernel)


                if random.random() < 0.7:
                    scale = random.uniform(0.125, 0.6)
                    h, w = img_lq.shape[:2]
                    img_lq = cv2.resize(img_lq, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

                    deg_info["resize_strength"] = (0.6 - scale) / (0.6 - 0.125)

                if random.random() < 0.8:
                    noise_level = random.uniform(2, 20)
                    noise = np.random.normal(0, noise_level, img_lq.shape).astype(np.float32)
                    img_lq = img_lq + noise

                    deg_info["noise_strength"] = (noise_level - 2.0) / (20.0 - 2.0)

                img_lq = np.clip(img_lq, 0, 255).astype(np.uint8)

                if random.random() < 0.8:
                    q_factor = random.randint(30, 95)
                    _, encimg = cv2.imencode('.jpg', img_lq, [int(cv2.IMWRITE_JPEG_QUALITY), q_factor])
                    img_lq = cv2.imdecode(encimg, 1)

                    deg_info["jpeg_strength"] = (95.0 - q_factor) / (95.0 - 30.0)


                img1_pil = Image.fromarray(cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB))
                expert_target = build_expert_target(deg_info)

            img0, img1 = self.pad_img([img0_pil, img1_pil], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1 = self.get_patch([img0, img1], self.image_size)

            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            images = [[img0, img1]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)

            # 🚀 核心修复：增加尺寸判断，杜绝原图尺寸已达标时的二次无谓插值
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
            else:
                return [
                    self.to_tensor(img0_out),
                    self.to_tensor(img1_out),
                    torch.tensor(expert_target, dtype=torch.float32)
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

            # 🚀 核心修复：增加尺寸判断
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

            # 🚀 核心修复：增加尺寸判断
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

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                # 🚀 补充修复：加上 sorted() 和 str()，确保双文件夹读取时的绝对顺序对齐！
                return sorted([str(p) for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')])

            if os.path.isfile(flist):
                try:
                    with open(flist, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines()]
                        lines = [l for l in lines if len(l) > 0]
                        return lines
                except Exception as e:
                    print(f"[Warning] Cannot read {flist} as text file. Treating as image path. Error: {e}")
                    return [flist]

        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
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
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h - patch_size)
        cc = random.randint(0, w - patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr + patch_size, cc:cc + patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
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
            bottom = bottom + (h // block_size) * block_size + \
                     (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                    (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

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
        bottom = bottom + (h // block_size) * block_size + \
                 (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]