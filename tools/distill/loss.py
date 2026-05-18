"""Feature-distillation loss.

Currently: per-stage channel-LN MSE + (1 - cosine similarity) on aligned
stages. Both terms reduce uniformly over spatial positions.

Saliency-weighted variant (planned, see docs/distill_saliency_weighted_loss.md):
accept an optional `saliency: Tensor (B, 1, H, W)` map from the teacher's
DBHead and use it to weight per-pixel contributions. This module is the
single point that needs to change when Option B lands.
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.distill.utils import channel_layernorm, cosine_sim_per_stage


@dataclass
class StageLoss:
    """Per-stage loss components, kept separate for logging."""
    mse: torch.Tensor               # scalar tensor (grad)
    cos_sim: float                  # detached scalar for logging
    # Saliency-weighted variant will add an `effective_pixels` count for
    # diagnostics. Not used yet.


def per_stage_distill_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    adapter: nn.Module,
    layernorm: bool = True,
) -> StageLoss:
    """Compute the per-stage MSE + cosine similarity.

    Args:
        student_feat: raw student stage output, (B, C^S, H, W)
        teacher_feat: raw teacher stage output, (B, C^T, H, W)
        adapter:      1x1 conv mapping C^S -> C^T
        layernorm:    apply per-token channel-wise LN before MSE (recommended;
                      ConvNeXt stage outputs are not LayerNormed)

    Returns:
        StageLoss with `mse` carrying the gradient and `cos_sim` detached.
    """
    s = adapter(student_feat)
    t = teacher_feat
    if layernorm:
        s_n = channel_layernorm(s)
        t_n = channel_layernorm(t)
        mse = F.mse_loss(s_n, t_n)
    else:
        mse = F.mse_loss(s, t)
    cos_sim = cosine_sim_per_stage(s.detach().float(), t.detach().float())
    return StageLoss(mse=mse, cos_sim=cos_sim)


def distill_loss(
    student_feats: List[torch.Tensor],
    teacher_feats: List[torch.Tensor],
    adapters: nn.ModuleList,
    align_stages: List[int],
    cosine_weight: float = 0.5,
    layernorm: bool = True,
):
    """Aggregate per-stage losses into a total scalar.

    Returns:
        total:         scalar tensor with gradient
        per_stage:     list[StageLoss] for logging
        loss_mse:      mean MSE across stages (tensor, for wandb)
        loss_cos:      mean (1 - cos) across stages (float, for wandb)
    """
    per_stage = []
    for adapter, stage_idx in zip(adapters, align_stages):
        per_stage.append(per_stage_distill_loss(
            student_feats[stage_idx], teacher_feats[stage_idx], adapter,
            layernorm=layernorm,
        ))
    loss_mse = sum(ps.mse for ps in per_stage) / len(per_stage)
    loss_cos = sum(1.0 - ps.cos_sim for ps in per_stage) / len(per_stage)
    total = loss_mse + cosine_weight * loss_cos
    return total, per_stage, loss_mse, loss_cos
