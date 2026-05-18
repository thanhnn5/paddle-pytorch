"""Teacher / student / adapter construction.

Two teacher variants share a unified `(features, prob_map_or_None)` forward:

- TeacherBackboneOnly: just the backbone. prob_map is None.
- TeacherWithSaliency: full BaseModel (backbone + LKPAN + DBHead). The DBHead
  sigmoid output IS the per-pixel text-probability map, fed back into
  loss.distill_loss as a saliency mask. See
  docs/distill_saliency_weighted_loss.md.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from torchocr import Config
from torchocr.modeling.architectures.base_model import BaseModel
from torchocr.modeling.backbones.det_convnext import ConvNeXtDetBackbone
from torchocr.modeling.backbones.det_convnextv2 import ConvNeXtV2Backbone
from torchocr.utils.logging import get_logger


class StageAdapter(nn.Module):
    """1x1 conv mapping student stage channels -> teacher stage channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        return self.proj(x)


class TeacherBackboneOnly(nn.Module):
    """Wraps a backbone so its forward matches TeacherWithSaliency's signature."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.out_channels = backbone.out_channels

    def forward(self, x) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        return list(self.backbone(x)), None


class TeacherWithSaliency(nn.Module):
    """Full BaseModel teacher; returns (stage features, per-pixel prob map).

    The prob map is the DBHead `res` output — already sigmoided in eval mode
    (see PFHeadLocal). Resolution matches the input.
    """

    def __init__(self, base: BaseModel):
        super().__init__()
        self.base = base
        self.out_channels = base.backbone.out_channels

    def forward(self, x) -> Tuple[List[torch.Tensor], torch.Tensor]:
        feats = self.base.backbone(x)
        neck_out = self.base.neck(feats)
        head_out = self.base.head(neck_out)
        if not isinstance(head_out, dict) or "res" not in head_out:
            raise RuntimeError(
                "Head did not return a dict with 'res'. Expected DBHead-family "
                f"head, got {type(head_out).__name__}."
            )
        prob_map = head_out["res"]
        # In some heads `res` carries (B, 3, H, W) during training (shrink +
        # threshold + binary stacked). In eval mode it's (B, 1, H, W). We're
        # always in eval here; take channel 0 to be safe.
        if prob_map.dim() == 4 and prob_map.size(1) > 1:
            prob_map = prob_map[:, :1]
        return list(feats), prob_map


def _split_state_dict(raw):
    """Unwrap nested checkpoint dicts to a flat state_dict."""
    if isinstance(raw, dict) and "model" in raw:
        return raw["model"]
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def build_teacher(
    ckpt_path: str,
    device: torch.device,
    *,
    full_model_config: Optional[str] = None,
) -> nn.Module:
    """Construct the teacher.

    Args:
        ckpt_path:          path to the BaseModel checkpoint.
        device:             where to place the teacher.
        full_model_config:  YAML path to the BaseModel architecture. If given,
                            build the full BaseModel + return
                            TeacherWithSaliency. If None, build the backbone
                            only.

    Both variants are eval-mode and have requires_grad=False on every param.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _split_state_dict(raw)
    logger = get_logger()

    if full_model_config is None:
        backbone = ConvNeXtDetBackbone(pretrained=True, finetune=False)
        backbone_state = {
            k[len("backbone."):]: v for k, v in state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        logger.info(
            f"teacher (backbone-only): matched {len(backbone_state)} keys, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        teacher: nn.Module = TeacherBackboneOnly(backbone)
    else:
        cfg = Config(full_model_config).cfg
        base = BaseModel(cfg["Architecture"])
        missing, unexpected = base.load_state_dict(state, strict=False)
        logger.info(
            f"teacher (full BaseModel from {full_model_config}): "
            f"loaded {len(state)} keys, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        teacher = TeacherWithSaliency(base)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.to(device)


def build_student(size: str, device: torch.device) -> ConvNeXtV2Backbone:
    student = ConvNeXtV2Backbone(size=size, pretrained=True, finetune=True)
    return student.to(device)


def build_adapters(student_ch, teacher_ch, align_stages, device) -> nn.ModuleList:
    """One 1x1 conv adapter per aligned stage."""
    return nn.ModuleList([
        StageAdapter(student_ch[s], teacher_ch[s]) for s in align_stages
    ]).to(device)
