"""Held-out feature-fidelity probe.

A fixed, deterministic, unaugmented subset of images, evaluated periodically
during training so different runs are comparable on the same metric (training-
batch metrics have too much aug noise for sweep comparison).
"""

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from tools.distill.dataset import UnlabeledImageFolder, build_eval_transform
from tools.distill.utils import autocast_for, channel_layernorm, cosine_sim_per_stage


def build_probe_batches(
    images_root: str,
    count: int,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    """Materialize a fixed set of probe batches. Same images on every call."""
    transform = build_eval_transform(img_size=img_size)
    ds = UnlabeledImageFolder(images_root, transform)
    indices = list(range(min(count, len(ds))))
    sub = Subset(ds, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    return [b for b in loader]


@torch.no_grad()
def run_probe(teacher, student, adapters, align_stages: List[int],
              probe_tensors, device: torch.device, amp: bool):
    """Returns per-stage mean MSE (raw + LN) and cosine similarity."""
    teacher.eval()
    student.eval()
    adapters.eval()

    mse_raw = [0.0] * len(align_stages)
    mse_ln = [0.0] * len(align_stages)
    cos = [0.0] * len(align_stages)
    n = 0
    for batch in probe_tensors:
        x = batch.to(device, non_blocking=True)
        with autocast_for(device, amp):
            t_feats, _prob_map = teacher(x)
            s_feats = student(x)
        bs = x.size(0)
        for i, (adapter, stage_idx) in enumerate(zip(adapters, align_stages)):
            # Adapter params are fp32; cast features up for numerical clarity.
            s = adapter(s_feats[stage_idx].float()).float()
            t = t_feats[stage_idx].float()
            mse_raw[i] += float(F.mse_loss(s, t).item()) * bs
            mse_ln[i] += float(F.mse_loss(channel_layernorm(s),
                                          channel_layernorm(t)).item()) * bs
            cos[i] += cosine_sim_per_stage(s, t) * bs
        n += bs

    student.train()
    adapters.train()
    return {
        "n": n,
        "mse_raw": [v / n for v in mse_raw],
        "mse_ln": [v / n for v in mse_ln],
        "cos": [v / n for v in cos],
    }
