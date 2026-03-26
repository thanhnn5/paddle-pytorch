"""
ConvNeXtDetBackbone: wraps DINOv3-ConvNeXt for use in PytorchOCR's BaseModel pipeline.

Uses facebook/dinov3-convnext-tiny-pretrain-lvd1689m from HuggingFace,
pretrained with DINOv3 self-supervised learning on LVD-1689M dataset.

Produces 4-scale feature maps at strides [4, 8, 16, 32] — compatible with
LKPAN and other 4-scale necks.  Pure convolution, no positional embeddings,
so train at 640x640 and inference at any resolution.

Architecture:
    Image -> ConvNeXtDetBackbone -> LKPAN -> PFHeadLocal (DBHead)
"""

import os

import torch.nn as nn

from transformers import AutoModel, AutoConfig


class ConvNeXtDetBackbone(nn.Module):
    """
    DINOv3-ConvNeXt backbone adapted for PytorchOCR detection.

    Outputs 4 feature maps (c2, c3, c4, c5) at strides [4, 8, 16, 32].

        backbone.out_channels -> list[int] consumed by Neck as in_channels
        backbone(x)           -> list of 4 tensors
    """

    _default_model = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"

    def __init__(
        self,
        in_channels: int = 3,                  # injected by BaseModel; unused
        model_name: str = None,
        weights_path: str = None,
        pretrained: bool = True,
        finetune: bool = True,
        hf_token: str = None,
    ):
        super().__init__()

        model_name = model_name or self._default_model
        token = hf_token or os.environ.get("HF_TOKEN")

        if weights_path and os.path.isdir(weights_path):
            # Load from local directory (already downloaded)
            self._backbone = AutoModel.from_pretrained(weights_path)
        elif pretrained:
            self._backbone = AutoModel.from_pretrained(model_name, token=token)
        else:
            config = AutoConfig.from_pretrained(model_name, token=token)
            self._backbone = AutoModel.from_config(config)

        self.out_channels = list(self._backbone.config.hidden_sizes)

        if not finetune:
            self._backbone.eval()
            self._backbone.requires_grad_(False)

    def forward(self, x):
        """Return list of 4 feature maps at strides [4, 8, 16, 32]."""
        out = self._backbone(x, output_hidden_states=True)
        # hidden_states: [input, stage1, stage2, stage3, stage4]
        # skip index 0 (original input), take stages 1-4
        features = out.hidden_states[1:]
        assert len(features) == 4, f"Expected 4 feature maps, got {len(features)}"
        return list(features)
