import os
import random
from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def convert_image_to_fn(img_type, image):
    if img_type is not None and image.mode != img_type:
        return image.convert(img_type)
    return image


class Dataset(Dataset):
    """
    Dataset used by RDDM restoration-style training.

    For BSR teacher training, use condition == 1 and pass folder as:
        folder[0] = HR flist
        folder[1] = LQ_up flist

    The returned order is:
        [HR, LQ_up]

    This order must not be changed, because ResidualDiffusion uses:
        x_start = imgs[0] = HR
        x_input = imgs[1] = LQ_up
        x_res   = x_input - x_start
    """

    def __init__(
        self,
        folder,
        image_size,
        exts=("jpg", "jpeg", "png", "bmp", "tif", "tiff"),
        augment_flip=False,
        convert_image_to=None,
        condition=0,
        equalizeHist=False,
        crop_patch=True,
        sample=False,
    ):
        super().__init__()

        self.equalizeHist = equalizeHist
        self.exts = tuple(e.lower().lstrip(".") for e in exts)
        self.augment_flip = augment_flip
        self.condition = condition
        self.crop_patch = crop_patch
        self.sample = sample
        self.image_size = image_size
        self.convert_image_to = convert_image_to

        if condition == 1:
            # Paired restoration / BSR teacher:
            # gt = HR, input = LQ_up
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self._check_list_length(self.gt, self.input, "gt", "input")

        elif condition == 0:
            # Unconditional generation
            self.paths = self.load_flist(folder)

        elif condition == 2:
            # Paired restoration with an extra condition map / mask
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self.input_condition = self.load_flist(folder[2])
            self._check_list_length(self.gt, self.input, "gt", "input")
            self._check_list_length(self.gt, self.input_condition, "gt", "input_condition")

        else:
            raise ValueError(f"Unsupported condition value: {condition}")

    def __len__(self):
        if self.condition:
            return len(self.input)
        return len(self.paths)

    def __getitem__(self, index):
        if self.condition == 1:
            # img0 = HR, img1 = LQ_up
            img0 = self._open_image(self.gt[index])
            img1 = self._open_image(self.input[index])
            self._check_pair_size([img0, img1], index)

            img0, img1 = self.pad_img([img0, img1], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1 = self.get_patch([img0, img1], self.image_size)

            if self.equalizeHist:
                img1 = self.cv2equalizeHist(img1)

            img0, img1 = self._augment_and_resize([img0, img1])
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            return [self.to_tensor(img0), self.to_tensor(img1)]

        if self.condition == 0:
            path = self.paths[index]
            img = self._open_image(path)

            img = self.pad_img([img], self.image_size)[0]

            if self.crop_patch and not self.sample:
                img = self.get_patch([img], self.image_size)[0]

            if self.equalizeHist:
                img = self.cv2equalizeHist(img)

            img = self._augment_and_resize([img])[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return self.to_tensor(img)

        if self.condition == 2:
            img0 = self._open_image(self.gt[index])
            img1 = self._open_image(self.input[index])
            img2 = self._open_image(self.input_condition[index])
            self._check_pair_size([img0, img1, img2], index)

            img0, img1, img2 = self.pad_img([img0, img1, img2], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1, img2 = self.get_patch([img0, img1, img2], self.image_size)

            if self.equalizeHist:
                img1 = self.cv2equalizeHist(img1)

            img0, img1, img2 = self._augment_and_resize([img0, img1, img2])
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            return [self.to_tensor(img0), self.to_tensor(img1), self.to_tensor(img2)]

        raise ValueError(f"Unsupported condition value: {self.condition}")

    def _open_image(self, path):
        path = os.fspath(path)
        img = Image.open(path)
        img = convert_image_to_fn(self.convert_image_to, img)
        return img

    def _check_list_length(self, a, b, name_a, name_b):
        if len(a) != len(b):
            raise ValueError(
                f"File list length mismatch: {name_a} has {len(a)} files, "
                f"but {name_b} has {len(b)} files."
            )

    def _check_pair_size(self, image_list, index):
        sizes = [img.size for img in image_list]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"Image size mismatch at index {index}: {sizes}. "
                "For RDDM BSR teacher training, HR and LQ_up must have the same size."
            )

    def load_flist(self, flist):
        """
        Supports:
            1. Python list / tuple / numpy array of file paths
            2. image directory
            3. text flist file, one image path per line
            4. a single image file
        """
        if isinstance(flist, (list, tuple, np.ndarray)):
            return [os.fspath(p) for p in flist]

        if not isinstance(flist, (str, os.PathLike)):
            return []

        flist = os.fspath(flist)
        path = Path(flist)

        if path.is_dir():
            paths = [
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower().lstrip(".") in self.exts
            ]
            return [str(p) for p in sorted(paths)]

        if path.is_file():
            # Single image file
            if path.suffix.lower().lstrip(".") in self.exts:
                return [str(path)]

            # Text flist file
            with open(path, "r", encoding="utf-8") as f:
                paths = [line.strip() for line in f.readlines() if line.strip()]
            return paths

        return []

    def cv2equalizeHist(self, img):
        # img is BGR numpy array here
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        # img is RGB numpy array here
        img = Image.fromarray(img)
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            name = os.fspath(self.input[index])

            if sub_dir == 0:
                return os.path.basename(name)

            if sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir_name = path.split("/")[-1]
                return sub_dir_name + "_" + os.path.basename(name)

            return os.path.basename(name)

        return f"{index}.png"

    def get_patch(self, image_list, patch_size):
        h, w = image_list[0].shape[:2]

        if h < patch_size or w < patch_size:
            raise ValueError(
                f"Image is smaller than patch size after padding: "
                f"image size=({h}, {w}), patch_size={patch_size}"
            )

        rr = random.randint(0, h - patch_size)
        cc = random.randint(0, w - patch_size)

        return [
            img[rr:rr + patch_size, cc:cc + patch_size, :].copy()
            for img in image_list
        ]

    def pad_img(self, img_list, patch_size, block_size=8):
        padded = []

        for img in img_list:
            # PIL RGB -> OpenCV BGR
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

            img = cv2.copyMakeBorder(
                img,
                0,
                bottom,
                0,
                right,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            padded.append(img)

        return padded

    def _augment_and_resize(self, image_list):
        # All images in image_list are BGR numpy arrays.
        if self.augment_flip and random.random() < 0.5:
            image_list = [cv2.flip(img, 1).copy() for img in image_list]

        # Original RDDM behavior:
        # if crop_patch is False, resize the full image to image_size.
        # For BSR teacher training, crop_patch=True is recommended.
        if not self.crop_patch:
            image_list = [
                cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                for img in image_list
            ]

        return image_list

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
