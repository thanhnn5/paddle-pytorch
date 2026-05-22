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


class HFImageDataset(Dataset):
    """Wrap a HuggingFace dataset for image-only Stage-2 distillation.

    First call to load_dataset downloads + caches the dataset to ~/.cache/
    huggingface/datasets (or HF_DATASETS_CACHE). Subsequent calls are local.

    Args:
        name:         HF repo id (e.g. "howard-hou/COCO-Text")
        transform:    albumentations transform applied to numpy RGB array
        splits:       list of splits to concatenate (default ['train']; use
                      ['train','validation'] to include both)
        image_key:    column name containing the image (both COCO-Text and
                      dlxjj/ICDAR2015 use 'image')
        cache_dir:    override HF cache location; useful for shared disks
        max_samples:  optional cap (truncate dataset; useful for smoke tests)
    """

    def __init__(self, name: str, transform, *, splits=("train",),
                 image_key: str = "image", cache_dir: str = None,
                 max_samples: int = None):
        from datasets import Image, concatenate_datasets, load_dataset
        loaded = []
        for split in splits:
            try:
                # verification_mode='no_checks' bypasses split-info mismatches
                # (e.g. dataset maintainer updates rows without updating the
                # README's YAML metadata, common on small HF datasets).
                loaded.append(load_dataset(
                    name, split=split, cache_dir=cache_dir,
                    verification_mode="no_checks",
                ))
            except Exception as e:
                # Some splits may be missing; warn but continue.
                print(f"[HFImageDataset] {name}: split={split!r} not loaded ({e})")
        if not loaded:
            raise RuntimeError(f"no splits loaded from {name}")
        self.ds = concatenate_datasets(loaded) if len(loaded) > 1 else loaded[0]
        if max_samples and max_samples < len(self.ds):
            self.ds = self.ds.select(range(max_samples))
        if image_key not in self.ds.column_names:
            raise RuntimeError(
                f"{name}: image_key={image_key!r} not in columns "
                f"{self.ds.column_names}"
            )
        # Cast image column to decode=False so we get {'bytes': ..., 'path': ...}
        # instead of an auto-decoded PIL backed by a now-closed zip file handle
        # (HF's lazy-decode breaks with concatenated datasets / multi-process
        # access). We decode ourselves in __getitem__.
        self.ds = self.ds.cast_column(image_key, Image(decode=False))
        self.transform = transform
        self.image_key = image_key
        self.name = name

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        from io import BytesIO
        from PIL import Image as PILImage
        sample = self.ds[idx]
        img_field = sample[self.image_key]
        # With decode=False, img_field is dict {'bytes': bytes|None, 'path': str|None}
        if isinstance(img_field, dict):
            if img_field.get("bytes") is not None:
                pil = PILImage.open(BytesIO(img_field["bytes"]))
            elif img_field.get("path"):
                # path can be a regular filesystem path OR an fsspec URI like
                # 'zip://img_1.jpg::/path/to/archive.zip' for zip-archived
                # datasets. Use fsspec.open to handle both.
                import fsspec
                with fsspec.open(img_field["path"], "rb") as f:
                    pil = PILImage.open(f)
                    pil.load()   # force read before file handle closes
            else:
                raise RuntimeError(f"empty image at {self.name}[{idx}]: {img_field}")
            img = np.array(pil.convert("RGB"))
        elif hasattr(img_field, "convert"):  # PIL (shouldn't happen post-cast)
            img = np.array(img_field.convert("RGB"))
        else:
            raise RuntimeError(
                f"unexpected image type {type(img_field).__name__} at {self.name}[{idx}]"
            )
        return self.transform(image=img)["image"]


class MixedImageDataset(Dataset):
    """Sample from multiple source datasets with configurable weights.

    Each __getitem__ picks a source by `weights`, then a random index within
    that source. Virtual length is `sum(len(s))` so one epoch sees roughly
    every image once on average (in expectation, weighted).

    Note: source weights are sampling probabilities per batch slot, not per-
    epoch proportions. If a small source is over-sampled (e.g. 1.5k ICDAR
    images at 20% weight in a 100k-step epoch), each ICDAR image will be
    revisited many times — that's intended for distillation (more exposure
    to scarce-but-distribution-valuable data).
    """

    def __init__(self, sources, weights, transform=None):
        assert len(sources) == len(weights) and len(sources) > 0
        self.sources = list(sources)
        w = np.asarray(weights, dtype=np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"weights must be non-negative with positive sum, got {weights}")
        self.weights = w / w.sum()
        self.transform = transform  # unused; per-source transforms still apply

    def __len__(self):
        return sum(len(s) for s in self.sources)

    def __getitem__(self, _idx):
        # Stochastic per-call (np.random is seeded per worker by seed_worker).
        src_idx = int(np.random.choice(len(self.sources), p=self.weights))
        src = self.sources[src_idx]
        sample_idx = int(np.random.randint(0, len(src)))
        return src[sample_idx]

    def describe(self):
        return [
            (getattr(s, "name", type(s).__name__), len(s), float(w))
            for s, w in zip(self.sources, self.weights)
        ]
