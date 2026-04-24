import os
import random
from pathlib import Path

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




def _random_iso_gaussian_kernel(kernel_size=21, sig_min=0.2, sig_max=4.0):
    """
    对齐 MCDFormer 的 iso_gaussian 模式：
    kernel_size = 21
    sigma ~ U(0.2, 4.0)
    """
    sigma = random.uniform(sig_min, sig_max)
    k1d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(k1d, k1d).astype(np.float32)
    kernel /= max(kernel.sum(), 1e-12)
    return kernel, sigma

def _fixed_iso_gaussian_kernel(kernel_size=21, sigma=0.0):
    """
    验证 / 测试阶段使用固定 sigma。
    sigma=0 表示 pure bicubic x4；
    sigma>0 表示 fixed iso Gaussian blur + bicubic x4。
    """
    sigma = float(sigma)
    if sigma <= 0:
        return None

    k1d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(k1d, k1d).astype(np.float32)
    kernel /= max(kernel.sum(), 1e-12)
    return kernel




def build_expert_target_sr_x4(sigma):
    """
    sigma soft target:
    E0: very mild / bicubic-like blur
    E1: sigma around 1.2
    E2: sigma around 2.4
    E3: sigma around 3.6

    注意：不要用 hard one-hot，因为 sigma 是连续变量。
    """
    sigma = float(np.clip(sigma, 0.0, 4.0))

    centers = np.array([0.4, 1.2, 2.4, 3.6], dtype=np.float32)
    widths = np.array([0.55, 0.60, 0.65, 0.70], dtype=np.float32)

    weights = np.exp(-0.5 * ((sigma - centers) / widths) ** 2).astype(np.float32)
    weights += 1e-4
    weights /= weights.sum()

    return weights.astype(np.float32)


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

        # 对齐 MCDFormer 的 x4 blind SR 设定
        self.sr_scale = 4
        self.blur_kernel_size = 21
        self.sig_min = 0.2
        self.sig_max = 4.0
        # 验证 / 测试固定退化强度
        # 由 train.py / test.py 通过环境变量传入
        # 0.0  -> pure bicubic x4
        # 1.2  -> Gaussian blur sigma=1.2 + bicubic x4
        # 2.4  -> Gaussian blur sigma=2.4 + bicubic x4
        # 3.6  -> Gaussian blur sigma=3.6 + bicubic x4
        self.eval_sigma = float(os.environ.get("MOE_EVAL_SIGMA", "0.0"))


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



    def degrade_for_eval_bicubic_x4(self, img_gt):
        """
        验证 / 测试对齐 MCDFormer:
        eval_sigma = 0:
            pure bicubic x4
        eval_sigma > 0:
            fixed iso Gaussian blur(eval_sigma) + bicubic x4

        当前 RDDM 条件输入要求和 GT 同尺寸，所以最后把 LR 再 bicubic 上采样回 GT 尺寸。
        """
        img = img_gt.astype(np.float32)
        h, w = img.shape[:2]

        # 1) fixed iso Gaussian blur
        kernel = _fixed_iso_gaussian_kernel(
            kernel_size=self.blur_kernel_size,
            sigma=self.eval_sigma
        )
        if kernel is not None:
            img = cv2.filter2D(img, -1, kernel)

        # 2) bicubic x4 downsample
        lr_w = max(1, w // self.sr_scale)
        lr_h = max(1, h // self.sr_scale)
        img_lr = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        # 3) quantize
        img_lr = np.clip(np.round(img_lr), 0, 255).astype(np.uint8)

        # 4) upsample back to GT size
        img_lq_up = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
        img_lq_up = np.ascontiguousarray(img_lq_up)

        return img_lq_up




    def degrade_for_blind_sr_x4(self, img_gt):
        """
        对齐 MCDFormer 的训练退化协议：
        1) iso gaussian blur, kernel_size=21, sigma in [0.2, 4.0]
        2) bicubic downsample x4
        3) quantize to uint8
        4) bicubic upsample back to GT size (为了兼容当前 RDDM 输入同尺寸条件)

        img_gt:
            HWC, uint8, BGR, 0~255

        return:
            img_lq_up: HWC, uint8, BGR, same size as GT
            expert_target: 4-dim soft label
        """
        img = img_gt.astype(np.float32)
        h, w = img.shape[:2]

        # 1) iso Gaussian blur
        kernel, sigma = _random_iso_gaussian_kernel(
            kernel_size=self.blur_kernel_size,
            sig_min=self.sig_min,
            sig_max=self.sig_max,
        )
        img = cv2.filter2D(img, -1, kernel)

        # 2) bicubic x4 downsample
        lr_w = max(1, w // self.sr_scale)
        lr_h = max(1, h // self.sr_scale)
        img_lr = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        # 3) quantize
        img_lr = np.clip(np.round(img_lr), 0, 255).astype(np.uint8)

        # 4) upsample back to GT size (for current RDDM condition input)
        img_lq_up = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
        img_lq_up = np.ascontiguousarray(img_lq_up)

        expert_target = build_expert_target_sr_x4(sigma)
        return img_lq_up, expert_target

    def __getitem__(self, index):
        if self.condition == 1:
            img0_pil = Image.open(self.gt[index])
            img0_pil = convert_image_to_fn(self.convert_image_to, img0_pil) if self.convert_image_to else img0_pil

            # ========== sample / inference 模式 ==========
            if self.sample:
                # 只从 GT/HR 读图，不再依赖外部 input 图
                img0 = self.pad_img([img0_pil], self.image_size)[0]

                # 验证 / 测试时不随机 crop，先 crop 到能被 scale 整除
                img0 = self.crop_to_scale(img0, self.sr_scale)

                # 在线生成 fixed sigma + bicubic x4 输入
                img1 = self.degrade_for_eval_bicubic_x4(img0)

                if self.equalizeHist:
                    img1 = self.cv2equalizeHist(img1)

                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

                return [self.to_tensor(img0), self.to_tensor(img1)]

            # ========== train 模式：先 HR 增强/crop，再退化 ==========
            img0 = self.pad_img([img0_pil], self.image_size)[0]

            # 对齐 MCDFormer：先 augment 再 crop
            if self.augment_flip:
                img0 = self.augment_img(img0)

            if self.crop_patch:
                img0 = self.get_patch([img0], self.image_size)[0]
            else:
                img0 = self.maybe_resize_to_square(img0, self.image_size)

            # 在 HR patch 上做退化
            img1, expert_target = self.degrade_for_blind_sr_x4(img0)

            if self.equalizeHist:
                img1 = self.cv2equalizeHist(img1)

            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            return [
                self.to_tensor(img0),
                self.to_tensor(img1),
                torch.tensor(expert_target, dtype=torch.float32),
            ]

        elif self.condition == 0:
            path = self.paths[index]
            img = Image.open(path)
            img = convert_image_to_fn(self.convert_image_to, img) if self.convert_image_to else img

            img = self.pad_img([img], self.image_size)[0]

            if self.augment_flip and not self.sample:
                img = self.augment_img(img)

            if self.crop_patch and not self.sample:
                img = self.get_patch([img], self.image_size)[0]
            else:
                img = self.maybe_resize_to_square(img, self.image_size)

            if self.equalizeHist:
                img = self.cv2equalizeHist(img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.to_tensor(img)

        elif self.condition == 2:
            img0 = Image.open(self.gt[index])
            img1 = Image.open(self.input[index])
            img2 = Image.open(self.input_condition[index])

            img0 = convert_image_to_fn(self.convert_image_to, img0) if self.convert_image_to else img0
            img1 = convert_image_to_fn(self.convert_image_to, img1) if self.convert_image_to else img1
            img2 = convert_image_to_fn(self.convert_image_to, img2) if self.convert_image_to else img2

            img0, img1, img2 = self.pad_img([img0, img1, img2], self.image_size)

            if self.augment_flip and not self.sample:
                img0, img1, img2 = self.augment_imgs([img0, img1, img2])

            if self.crop_patch and not self.sample:
                img0, img1, img2 = self.get_patch([img0, img1, img2], self.image_size)
            else:
                img0 = self.maybe_resize_to_square(img0, self.image_size)
                img1 = self.maybe_resize_to_square(img1, self.image_size)
                img2 = self.maybe_resize_to_square(img2, self.image_size)

            if self.equalizeHist:
                img1 = self.cv2equalizeHist(img1)

            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

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

    def augment_img(self, img):
        """
        对齐 MCDFormer 的增强风格：
        hflip / vflip / rot90
        输入输出均为 HWC, BGR
        """
        if random.random() < 0.5:
            img = img[:, ::-1, :]
        if random.random() < 0.5:
            img = img[::-1, :, :]
        if random.random() < 0.5:
            img = img.transpose(1, 0, 2)
        return np.ascontiguousarray(img)

    def augment_imgs(self, image_list):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        out = []
        for img in image_list:
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
            out.append(np.ascontiguousarray(img))
        return out

    def get_patch(self, image_list, patch_size):
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h - patch_size)
        cc = random.randint(0, w - patch_size)

        out = []
        for img in image_list:
            out.append(img[rr:rr + patch_size, cc:cc + patch_size, :])
        return out

    def maybe_resize_to_square(self, img, target_size):
        h, w = img.shape[:2]
        if h != target_size or w != target_size:
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        return img

    def crop_to_scale(self, img, scale=4):
        """
        对齐 MCDFormer 的 crop_border：
        在验证 / 测试退化前，先把 HR crop 到能被 scale 整除。
        """
        h, w = img.shape[:2]
        h = int(h // scale * scale)
        w = int(w // scale * scale)
        return img[:h, :w, :]

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
        path = self.gt[index] if hasattr(self, "gt") else self.input[index]
        img = Image.open(path)
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