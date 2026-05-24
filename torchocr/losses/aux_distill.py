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
    """Combined feature + logit distillation loss.

    Two complementary signals (either or both can be enabled via weights):

    * Feature distillation (weight = `feat_weight`): per-stage cosine-similarity
      between adapter-projected student backbone features and teacher backbone
      features. Pulls the student's backbone representation toward the
      teacher's.

    * Logit distillation (weight = `logit_weight`): MSE between student and
      teacher per-pixel text-probability maps (the DBHead-family `res[:, :1]`
      sigmoid output). Dense supervision over ALL pixels — especially the
      ~85-95% unlabeled background where detection labels are silent. This is
      what closes the precision gap (the teacher's "this looks like text but
      isn't" calibration is dark knowledge labels can't provide).

    Args:
        student_channels: full list of student backbone stage channel dims
        teacher_channels: full list of teacher backbone stage channel dims
        align_stages:     0-indexed backbone stages to align for feature
                          distillation (e.g. [2, 3])
        feat_weight:      multiplier on the feature distillation term. 0
                          disables it.
        logit_weight:     multiplier on the logit distillation term. 0
                          disables it (default — feature-only, current
                          behavior).

    Forward returns a dict with keys:
        loss        : the weighted sum (this is what the trainer adds to
                      detection loss); has gradient
        loss_feat   : unweighted feature loss for logging; detached
        loss_logit  : unweighted logit loss for logging; detached (only
                      present when logit_weight > 0 and prob maps provided)
    """

    def __init__(
        self,
        student_channels: Sequence[int],
        teacher_channels: Sequence[int],
        align_stages: Sequence[int],
        feat_weight: float = 5.0,
        logit_weight: float = 0.0,
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
        self.feat_weight = float(feat_weight)
        self.logit_weight = float(logit_weight)
        # Backward-compat alias (older trainer code referenced `.weight`).
        self.weight = self.feat_weight
        # 1x1 conv adapter per aligned stage. No bias (RT-DETRv4 style).
        self.adapters = nn.ModuleList([
            nn.Conv2d(student_channels[s], teacher_channels[s], kernel_size=1, bias=False)
            for s in self.align_stages
        ])

    def _feature_loss(
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
            if s.shape[-2:] != t.shape[-2:]:
                t = F.interpolate(t, size=s.shape[-2:],
                                  mode="bilinear", align_corners=False)
            s_proj = adapter(s.float())
            t_f = t.float()
            s_flat = s_proj.flatten(2).permute(0, 2, 1).contiguous()
            t_flat = t_f.flatten(2).permute(0, 2, 1).contiguous()
            cos = F.cosine_similarity(s_flat, t_flat, dim=-1)
            losses.append((1.0 - cos).mean())
        return torch.stack(losses).mean()

    def _logit_loss(
        self,
        student_prob: torch.Tensor,
        teacher_prob: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between sigmoid prob maps (single channel).

        Both inputs `.contiguous()`-ified — caller-side slices (e.g.
        `preds['res'][:, :1]`) produce non-contiguous views that some
        backends (notably MPS) refuse to backprop through.
        """
        s = student_prob.float()
        t = teacher_prob.float()
        if s.shape[-2:] != t.shape[-2:]:
            t = F.interpolate(t, size=s.shape[-2:],
                              mode="bilinear", align_corners=False)
        if s.size(1) > 1:
            s = s[:, :1]
        if t.size(1) > 1:
            t = t[:, :1]
        return F.mse_loss(s.contiguous(), t.contiguous())

    def forward(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
        student_prob: torch.Tensor = None,
        teacher_prob: torch.Tensor = None,
    ) -> dict:
        out = {}
        total = 0.0
        any_term = False

        if self.feat_weight > 0:
            l_feat = self._feature_loss(student_feats, teacher_feats)
            out['loss_feat'] = l_feat.detach()
            total = total + self.feat_weight * l_feat
            any_term = True

        if self.logit_weight > 0 and student_prob is not None and teacher_prob is not None:
            l_logit = self._logit_loss(student_prob, teacher_prob)
            out['loss_logit'] = l_logit.detach()
            total = total + self.logit_weight * l_logit
            any_term = True

        if not any_term:
            # Edge case: both weights are 0 or prob maps missing when needed.
            # Return zero so caller can still proceed; trainer skips this.
            device = student_feats[0].device if student_feats else torch.device('cpu')
            total = torch.zeros((), device=device, requires_grad=True)

        out['loss'] = total
        return out
