"""Feature-distillation loss.

Per-stage channel-LN MSE + (1 - cosine similarity) on aligned stages.

When `saliency` is given, the MSE is weighted per-pixel by
`alpha + beta * saliency_at_stage_resolution`. This focuses the gradient on
text regions (the teacher's DBHead probability map), which matters when the
target (the address) is a small fraction of each crop. See
docs/distill_saliency_weighted_loss.md.

The cosine loss stays unweighted by default since it's already
spatially-averaged and small.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.distill.utils import channel_layernorm, cosine_sim_per_stage


@dataclass
class StageLoss:
    mse: torch.Tensor               # scalar tensor (with grad)
    cos_sim: float                  # detached scalar for logging


def _resize_saliency(saliency: torch.Tensor, shape) -> torch.Tensor:
    """Bilinear-downsample saliency to the given (H, W)."""
    return F.interpolate(saliency, size=shape, mode="bilinear", align_corners=False)


def per_stage_distill_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    adapter: nn.Module,
    *,
    layernorm: bool = True,
    saliency: Optional[torch.Tensor] = None,
    sal_alpha: float = 0.1,
    sal_beta: float = 1.0,
) -> StageLoss:
    """One stage's MSE + cos. With saliency, MSE is per-pixel weighted."""
    s = adapter(student_feat)
    t = teacher_feat
    if layernorm:
        s_for_mse = channel_layernorm(s)
        t_for_mse = channel_layernorm(t)
    else:
        s_for_mse, t_for_mse = s, t

    if saliency is None:
        mse = F.mse_loss(s_for_mse, t_for_mse)
    else:
        # saliency: (B, 1, H_in, W_in) -> resize to stage spatial size
        w = _resize_saliency(saliency, s.shape[-2:])  # (B, 1, H_s, W_s)
        weight = sal_alpha + sal_beta * w             # (B, 1, H_s, W_s)
        diff_sq = (s_for_mse - t_for_mse) ** 2        # (B, C, H_s, W_s)
        # Per-pixel weighted mean: sum(weight * diff_sq) / (sum(weight) * C)
        num = (diff_sq * weight).sum()
        den = weight.sum() * s_for_mse.size(1)        # weight.sum() counts B*H*W; mul by C
        mse = num / den.clamp_min(1e-8)

    cos_sim = cosine_sim_per_stage(s.detach().float(), t.detach().float())
    return StageLoss(mse=mse, cos_sim=cos_sim)


def distill_loss(
    student_feats: List[torch.Tensor],
    teacher_feats: List[torch.Tensor],
    adapters: nn.ModuleList,
    align_stages: List[int],
    *,
    cosine_weight: float = 0.5,
    layernorm: bool = True,
    saliency: Optional[torch.Tensor] = None,
    sal_alpha: float = 0.1,
    sal_beta: float = 1.0,
):
    """Aggregate per-stage losses into a total scalar.

    Returns:
        total:     scalar tensor with gradient
        per_stage: list[StageLoss] (per-stage MSE tensor + detached cosine)
        loss_mse:  mean MSE across stages (tensor, for logging)
        loss_cos:  mean (1 - cos) across stages (float, for logging)
    """
    per_stage = []
    for adapter, stage_idx in zip(adapters, align_stages):
        per_stage.append(per_stage_distill_loss(
            student_feats[stage_idx], teacher_feats[stage_idx], adapter,
            layernorm=layernorm,
            saliency=saliency, sal_alpha=sal_alpha, sal_beta=sal_beta,
        ))
    loss_mse = sum(ps.mse for ps in per_stage) / len(per_stage)
    loss_cos = sum(1.0 - ps.cos_sim for ps in per_stage) / len(per_stage)
    total = loss_mse + cosine_weight * loss_cos
    return total, per_stage, loss_mse, loss_cos
