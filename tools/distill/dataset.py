"""Unlabeled image folder dataset for ConvNeXt feature distillation.

Aggressive augmentation per `docs/convnext_distillation.md` Option C: strong
photometric + mild geometric, NO mosaic/cutmix (those scramble document
structure the teacher's features depend on).
"""

import glob
import os

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = 640):
    """Aggressive aug for POD-only distillation."""
    return A.Compose([
        A.LongestMaxSize(max_size=int(img_size * 1.4), interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=int(img_size * 1.4), min_width=int(img_size * 1.4),
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0),
                            ratio=(0.85, 1.18), interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), translate_percent=(-0.05, 0.05),
                 scale=(0.9, 1.1), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02, p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 1.5), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.ImageCompression(quality_range=(50, 100), p=0.5),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(img_size: int = 640):
    """Deterministic transform for held-out feature-fidelity probes."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _collect_paths(root: str, exts=("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root, f"**/*.{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(root, f"**/*.{ext.upper()}"), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError(f"no images found under {root}")
    return paths


class UnlabeledImageFolder(Dataset):
    def __init__(self, root: str, transform):
        self.paths = _collect_paths(root)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            # cv2 fails on some corrupted/odd files; skip by returning the next one
            return self.__getitem__((idx + 1) % len(self.paths))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"]
