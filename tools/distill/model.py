"""Teacher / student / adapter construction.

The current `build_teacher` loads only the backbone. The saliency-weighted
loss plan (docs/distill_saliency_weighted_loss.md) will replace this with a
wrapper that also exposes the DBHead probability map.
"""

import torch
import torch.nn as nn

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


def build_teacher(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load the task-adapted DINOv3-ConvNeXt-tiny backbone from a BaseModel ckpt.

    The checkpoint is the full BaseModel state_dict (backbone + neck + head);
    we strip the `backbone.` prefix and load only those weights. The neck +
    head weights are dropped because Stage 2 only needs feature maps.

    To switch to saliency-weighted distillation later, build the full
    BaseModel here instead and return a wrapper that exposes (features,
    prob_map). See docs/distill_saliency_weighted_loss.md.
    """
    teacher = ConvNeXtDetBackbone(pretrained=True, finetune=False)
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    backbone_state = {}
    for k, v in raw.items():
        if k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
    missing, unexpected = teacher.load_state_dict(backbone_state, strict=False)
    get_logger().info(
        f"teacher load: matched {len(backbone_state)} keys, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
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
