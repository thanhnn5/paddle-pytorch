"""
DINOv3DetBackbone: wraps DEIMv2's DINOv3STAs for use in PytorchOCR's BaseModel pipeline.

Architecture:
    Image -> DINOv3DetBackbone -> HybridEncoderNeck -> PFHeadLocal (DBHead)
"""

import os

import torch
import torch.nn as nn

from torchocr.modeling.thirdparty.deimv2.backbone.dinov3_adapter import DINOv3STAs


# ---------------------------------------------------------------------------
# Helper: fix NaN bias_mask in LinearKMaskedBias layers
# ---------------------------------------------------------------------------

def _fix_nan_bias_mask(module: nn.Module) -> None:
    """
    Replace NaN-initialised bias_mask buffers in LinearKMaskedBias layers with 1.0.

    DINOv3's LinearKMaskedBias registers bias_mask=NaN as a sentinel that is
    overwritten when loading pretrained weights.  When training from scratch the
    mask stays NaN and propagates through the attention computation.  Setting it
    to 1.0 makes the layer behave identically to a plain nn.Linear (K-bias is
    used as-is, no masking), which is the safe default for random initialisation.
    """
    for buf_name, buf in list(module.named_buffers(recurse=False)):
        if buf_name == "bias_mask" and buf is not None and torch.isnan(buf).all():
            buf.fill_(1.0)
    for child in module.children():
        _fix_nan_bias_mask(child)


# ---------------------------------------------------------------------------
# Helper: convert SyncBatchNorm -> BatchNorm2d (needed for single-GPU runs)
# ---------------------------------------------------------------------------

def _syncbn_to_bn(module: nn.Module) -> None:
    """Recursively replace all nn.SyncBatchNorm children with nn.BatchNorm2d."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            if child.affine:
                bn.weight = nn.Parameter(child.weight.data.clone())
                bn.bias = nn.Parameter(child.bias.data.clone())
            if child.track_running_stats:
                bn.running_mean.copy_(child.running_mean)
                bn.running_var.copy_(child.running_var)
                bn.num_batches_tracked.copy_(child.num_batches_tracked)
            setattr(module, name, bn)
        else:
            _syncbn_to_bn(child)


# ---------------------------------------------------------------------------
# Main backbone wrapper
# ---------------------------------------------------------------------------

class DINOv3DetBackbone(nn.Module):
    """
    Thin wrapper around DEIMv2's DINOv3STAs that conforms to PytorchOCR's
    BaseModel interface:

        backbone.out_channels  -> list[int] consumed by Neck as in_channels
        backbone(x)            -> tuple (c2, c3, c4) at strides [8, 16, 32]
    """

    def __init__(
        self,
        in_channels: int = 3,               # injected by BaseModel; unused internally
        name_variant: str = "dinov3_vits16",  # passed as 'name' to DINOv3STAs
        weights_path=None,
        interaction_indexes=None,
        hidden_dim: int = 256,
        conv_inplane: int = 32,
        use_sta: bool = True,
        finetune: bool = True,
        sync_bn: bool = False,              # True only under torch DDP
    ):
        super().__init__()

        if interaction_indexes is None:
            interaction_indexes = [5, 8, 11]

        self._backbone = DINOv3STAs(
            name=name_variant,
            weights_path=weights_path,
            interaction_indexes=interaction_indexes,
            hidden_dim=hidden_dim,
            conv_inplane=conv_inplane,
            use_sta=use_sta,
            finetune=finetune,
        )

        if not sync_bn:
            _syncbn_to_bn(self._backbone)

        # When no pretrained weights are loaded the LinearKMaskedBias layers in
        # DINOv3 leave their bias_mask buffer as NaN (a sentinel for checkpoint
        # loading).  This causes NaN propagation during forward passes.  Replace
        # any remaining NaN masks with 1.0 so the layer acts as plain nn.Linear.
        if weights_path is None or not os.path.exists(weights_path):
            _fix_nan_bias_mask(self._backbone)

        # PytorchOCR BaseModel reads this to set Neck's in_channels
        self.out_channels = [hidden_dim, hidden_dim, hidden_dim]

    def forward(self, x):
        """Return (c2, c3, c4) feature maps at strides 8, 16, 32."""
        out = self._backbone(x)
        assert len(out) == 3, f"Expected 3 feature maps from backbone, got {len(out)}"
        return out
