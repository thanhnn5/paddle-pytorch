"""ConvNeXt-V2 femto/atto student backbone for distillation + detection.

Mirrors the 4-stage output interface of ConvNeXtDetBackbone (strides 4/8/16/32)
so it drops into the same LKPAN/DBHead pipeline.
"""

import os

import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ConvNeXtV2Backbone(nn.Module):
    """Compact ConvNeXt-V2 student.

    out_channels: list[int] consumed by Neck as in_channels.
    forward(x): list of 4 tensors at strides [4, 8, 16, 32].
    """

    _size_to_model = {
        "atto":  "facebook/convnextv2-atto-1k-224",
        "femto": "facebook/convnextv2-femto-1k-224",
        "pico":  "facebook/convnextv2-pico-1k-224",
        "nano":  "facebook/convnextv2-nano-1k-224",
        "tiny":  "facebook/convnextv2-tiny-1k-224",
    }

    def __init__(
        self,
        in_channels: int = 3,                  # injected by BaseModel; unused
        size: str = "femto",
        model_name: str = None,
        weights_path: str = None,
        pretrained: bool = True,
        finetune: bool = True,
        hf_token: str = None,
    ):
        super().__init__()

        if model_name is None:
            if size not in self._size_to_model:
                raise ValueError(f"unknown size {size!r}, choose from {list(self._size_to_model)}")
            model_name = self._size_to_model[size]

        token = hf_token or os.environ.get("HF_TOKEN")

        if weights_path and os.path.isdir(weights_path):
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
        # ConvNextV2Model emits 5 hidden states: [embeddings, stage1, ..., stage4].
        # Take stages 1-4 to match the 4-stride interface.
        features = out.hidden_states[1:]
        assert len(features) == 4, f"Expected 4 feature maps, got {len(features)}"
        return list(features)
