"""Cross-cutting helpers: seeding, device selection, autocast, feature ops."""

import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs so trials are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """DataLoader worker_init_fn.

    Each worker derives a deterministic RNG from torch.initial_seed(), so
    albumentations (numpy) augmentation order is reproducible across runs
    that share the same --seed.
    """
    base = torch.initial_seed() % (2**32)
    np.random.seed(base)
    random.seed(base)


def enable_deterministic_mode():
    """Opt into bit-exact reproducibility on CUDA (10-30% throughput cost).

    MPS retains hardware-level atomic nondeterminism that this cannot fix.
    """
    import os
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_for(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        # MPS bf16 is supported on torch >= 2.3 but flaky for some ops; use fp16.
        return torch.amp.autocast("mps", dtype=torch.float16)
    return nullcontext()


def channel_layernorm(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-token channel-wise LN: normalize each spatial position over channels."""
    mean = feat.mean(dim=1, keepdim=True)
    var = feat.var(dim=1, keepdim=True, unbiased=False)
    return (feat - mean) / torch.sqrt(var + eps)


def cosine_sim_per_stage(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> float:
    """Mean cosine similarity across spatial positions, per batch sample."""
    s = F.normalize(student_feat.flatten(2), dim=1)
    t = F.normalize(teacher_feat.flatten(2), dim=1)
    return (s * t).sum(dim=1).mean().item()
