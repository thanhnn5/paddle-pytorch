"""
HybridEncoderNeck: wraps DEIMv2's HybridEncoder for use in PytorchOCR's BaseModel pipeline.

Architecture:
    (c2, c3, c4) @ strides [8, 16, 32]
        -> HybridEncoder  -> [p2, p3, p4] @ strides [8, 16, 32]
        -> merge + 2x upsample             -> [B, hidden_dim, H/4, W/4]
        -> merge_conv                      -> [B, out_channels, H/4, W/4]
"""

import torch.nn as nn
import torch.nn.functional as F

from torchocr.modeling.thirdparty.deimv2.deim.hybrid_encoder import HybridEncoder


# ---------------------------------------------------------------------------
# Main neck wrapper
# ---------------------------------------------------------------------------

class HybridEncoderNeck(nn.Module):
    """
    Thin wrapper around DEIMv2's HybridEncoder that conforms to PytorchOCR's
    BaseModel neck interface:

        self.out_channels  -> int, consumed by DBHead/PFHeadLocal as in_channels
        forward(feats)     -> single tensor [B, out_channels, H/4, W/4]
    """

    def __init__(
        self,
        in_channels,                    # list [hd, hd, hd] injected by BaseModel
        out_channels: int = 256,
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_encoder_layers: int = 1,
        use_encoder_idx=None,            # default: [2]
        feat_strides=None,              # default: [8, 16, 32]
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        version: str = "deim",
    ):
        super().__init__()

        if use_encoder_idx is None:
            use_encoder_idx = [2]
        if feat_strides is None:
            feat_strides = [8, 16, 32]

        assert all(c == hidden_dim for c in in_channels), (
            f"All in_channels must equal hidden_dim={hidden_dim}, got {in_channels}"
        )

        self.out_channels = out_channels  # single int consumed by head

        self.encoder = HybridEncoder(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            expansion=expansion,
            depth_mult=depth_mult,
            version=version,
            fuse_op="sum",
        )

        self.merge_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        """
        Args:
            feats: tuple/list of 3 tensors (c2, c3, c4) at strides [8, 16, 32]

        Returns:
            Tensor [B, out_channels, H/4, W/4]
        """
        encoder_out = self.encoder(list(feats))  # [p2@1/8, p3@1/16, p4@1/32]
        assert len(encoder_out) == 3, f"HybridEncoder returned {len(encoder_out)} feature maps, expected 3"
        p2, p3, p4 = encoder_out

        p3_up = F.interpolate(p3, size=p2.shape[2:], mode="nearest")
        p4_up = F.interpolate(p4, size=p2.shape[2:], mode="nearest")
        merged = p2 + p3_up + p4_up  # [B, hidden_dim, H/8, W/8]

        out = F.interpolate(merged, scale_factor=2, mode="nearest")  # [B, hidden_dim, H/4, W/4]
        return self.merge_conv(out)  # [B, out_channels, H/4, W/4]
