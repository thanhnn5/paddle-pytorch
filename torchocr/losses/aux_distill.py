"""Stage-3 auxiliary distillation loss.

Cosine-similarity-based feature alignment between the student's backbone
stage outputs and a frozen teacher's matching stage outputs, with per-stage
1x1 conv adapters to bridge channel-dim mismatches.

Inspired by RT-DETRv4's Dense Spatial Imitation (DSI) loss. See
docs/stage3_auxiliary_distill.md for the full design rationale.

This module only owns the *adapters* and the *loss computation*. The teacher
is managed externally by the trainer (loaded once, frozen, kept off the
model.state_dict() so it doesn't bloat checkpoints, and excluded from the
optimizer).
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxDistillLoss(nn.Module):
    """Per-stage cosine-similarity distillation loss with channel adapters.

    Args:
        student_channels: full list of student backbone stage channel dims
                          (e.g. [48, 96, 192, 384] for femto).
        teacher_channels: full list of teacher backbone stage channel dims
                          (e.g. [96, 192, 384, 768] for ConvNeXt-tiny).
        align_stages:     0-indexed stage indices to align (e.g. [2, 3] for
                          strides 16 and 32).
        weight:           multiplier on the distillation loss before adding
                          to the detection loss. Read by the trainer.

    Forward signature: (student_feats, teacher_feats) -> scalar loss tensor.
        student_feats: list of 4 stage tensors from student.backbone
        teacher_feats: list of 4 stage tensors from teacher.backbone
        Returns: scalar with gradient w.r.t. the student feats (and adapter
                 params).
    """

    def __init__(
        self,
        student_channels: Sequence[int],
        teacher_channels: Sequence[int],
        align_stages: Sequence[int],
        weight: float = 5.0,
    ):
        super().__init__()
        if not align_stages:
            raise ValueError("align_stages must be non-empty")
        for s in align_stages:
            if not (0 <= s < len(student_channels)):
                raise ValueError(f"align_stage {s} out of range for "
                                 f"{len(student_channels)} student stages")
            if not (0 <= s < len(teacher_channels)):
                raise ValueError(f"align_stage {s} out of range for "
                                 f"{len(teacher_channels)} teacher stages")

        self.align_stages = list(align_stages)
        self.weight = float(weight)
        # 1x1 conv adapter per aligned stage. No bias (RT-DETRv4 style).
        self.adapters = nn.ModuleList([
            nn.Conv2d(student_channels[s], teacher_channels[s], kernel_size=1, bias=False)
            for s in self.align_stages
        ])

    def forward(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(student_feats) <= max(self.align_stages):
            raise RuntimeError(
                f"student_feats has {len(student_feats)} stages but "
                f"align_stages references index {max(self.align_stages)}"
            )
        if len(teacher_feats) <= max(self.align_stages):
            raise RuntimeError(
                f"teacher_feats has {len(teacher_feats)} stages but "
                f"align_stages references index {max(self.align_stages)}"
            )

        losses = []
        for adapter, stage_idx in zip(self.adapters, self.align_stages):
            s = student_feats[stage_idx]
            t = teacher_feats[stage_idx]

            # Spatial size may differ if e.g. teacher was processed at a
            # different resolution; resize teacher to match student.
            if s.shape[-2:] != t.shape[-2:]:
                t = F.interpolate(t, size=s.shape[-2:],
                                  mode="bilinear", align_corners=False)

            # Adapter runs in fp32 even under autocast to keep cos-sim stable.
            s_proj = adapter(s.float())   # (B, C_t, H, W)
            t_f = t.float()

            # Flatten spatial -> token sequence (B, N, C_t)
            s_flat = s_proj.flatten(2).permute(0, 2, 1).contiguous()
            t_flat = t_f.flatten(2).permute(0, 2, 1).contiguous()

            cos = F.cosine_similarity(s_flat, t_flat, dim=-1)  # (B, N)
            losses.append((1.0 - cos).mean())

        return torch.stack(losses).mean()
