"""PPLCNetV4 detection backbone (PP-OCRv6 det), PyTorch port.

Ported from PaddleOCR ``ppocr/modeling/backbones/rec_lcnetv4.py`` (det branch
only). Module / parameter names mirror the Paddle implementation one-to-one so a
PaddleOCR ``.pdparams`` converts with a pure rename (``_mean``->``running_mean``,
``_variance``->``running_var``); see tools/deployment/convert_paddle_weights.py.

Paddle-vs-torch details that matter for numerical parity:
  * The stem's 2x2 convs and the 2x2 max-pool use Paddle ``padding="SAME"`` with
    stride 1, i.e. asymmetric (0 before, 1 after) padding. Done with an explicit
    ``F.pad`` so torch.export / CoreML see a plain conv/pool.
  * The max-pool input is post-ReLU (>= 0), so zero padding is exact (Paddle's
    SAME pool ignores the padded cells).
  * ``nn.Hardsigmoid`` in Paddle defaults to slope 1/6, same as torch.

``rep()`` (called by tools/deployment/export_*.py) merges RepDWConv's
3x3 + 1x1 + identity branches into a single 3x3 depthwise conv.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PPLCNetV4"]

# [kernel, in_c, out_c, stride, use_se]
NET_CONFIG_DET = {
    "tiny": {
        "stem": (16, 32),
        "blocks_s1": [[3, 32, 32, 1, True], [3, 32, 32, 1, False]],
        "blocks_s2": [[3, 32, 48, 2, False], [3, 48, 48, 1, True], [3, 48, 48, 1, False]],
        "blocks_s3": [
            [3, 48, 64, 2, False],
            [3, 64, 64, 1, True],
            [3, 64, 64, 1, False],
            [3, 64, 64, 1, True],
            [3, 64, 64, 1, False],
        ],
        "blocks_s4": [[3, 64, 160, 2, False], [3, 160, 160, 1, True], [3, 160, 160, 1, False]],
    },
    "small": {
        "stem": (24, 48),
        "blocks_s1": [[3, 48, 48, 1, True], [3, 48, 48, 1, False]],
        "blocks_s2": [[3, 48, 96, 2, False], [3, 96, 96, 1, True], [3, 96, 96, 1, False]],
        "blocks_s3": [
            [3, 96, 192, 2, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
        ],
        "blocks_s4": [[3, 192, 384, 2, False], [3, 384, 384, 1, True], [3, 384, 384, 1, False]],
    },
    "medium": {
        "stem": (64, 128),
        "blocks_s1": [[3, 128, 128, 1, True], [3, 128, 128, 1, False]],
        "blocks_s2": [[3, 128, 256, 2, False], [3, 256, 256, 1, True], [3, 256, 256, 1, False]],
        "blocks_s3": [
            [3, 256, 512, 2, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
        ],
        "blocks_s4": [[3, 512, 896, 2, False], [3, 896, 896, 1, True], [3, 896, 896, 1, False]],
    },
}


def _fuse_conv_bn_weights(conv_w, conv_b, bn):
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std
    w = conv_w * scale.reshape(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * scale
    if conv_b is not None:
        b = b + conv_b * scale
    return w, b


class Conv2D_BN(nn.Sequential):
    """Conv (no bias) + BN. Keys: ``conv.weight``, ``bn.*``."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                               groups=groups, bias=False)),
            ("bn", nn.BatchNorm2d(out_channels)),
        ]))


class ConvBNAct(nn.Module):
    """Conv + BN + ReLU. ``same_pad=True`` reproduces Paddle ``padding="SAME"``
    (stride 1): total pad k-1 split as (k-1)//2 before, rest after."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, same_pad=False, use_act=True):
        super().__init__()
        self.use_act = use_act
        if same_pad:
            assert stride == 1, "SAME padding only ported for stride 1"
            total = kernel_size - 1
            before = total // 2
            self.pad = (before, total - before, before, total - before)
            padding = 0
        else:
            self.pad = None
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        if self.pad is not None:
            x = F.pad(x, self.pad)
        return self.act(self.bn(self.conv(x)))


class StemBlock(nn.Module):
    """Multi-branch stem, total stride 4."""

    def __init__(self, in_channels=3, mid_channels=48, out_channels=96):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, mid_channels, 3, 2)
        self.stem2a = ConvBNAct(mid_channels, mid_channels // 2, 2, 1, same_pad=True)
        self.stem2b = ConvBNAct(mid_channels // 2, mid_channels, 2, 1, same_pad=True)
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, 3, 2)
        self.stem4 = ConvBNAct(mid_channels, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2b(self.stem2a(x))
        # Paddle MaxPool2D(2, 1, padding="SAME"): pad (0, 1). x >= 0 (post-ReLU) so
        # zero padding never wins the max -> exact.
        x1 = self.pool(F.pad(x, (0, 1, 0, 1)))
        return self.stem4(self.stem3(torch.cat([x1, x2], dim=1)))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = self.hardsigmoid(self.conv2(self.relu(self.conv1(s))))
        return x * s


class RepDWConv(nn.Module):
    """Train: BN(DW3x3_BN(x) + DW1x1(x) + x). Deploy (after rep): one DW3x3."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.conv = Conv2D_BN(channels, channels, kernel_size, 1, padding, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.is_repped = False

    def forward(self, x):
        if self.is_repped:
            return self.reparam_conv(x)
        return self.bn(self.conv(x) + self.conv1(x) + x)

    @torch.no_grad()
    def rep(self):
        if self.is_repped:
            return
        k = self.kernel_size
        p = k // 2
        w, b = _fuse_conv_bn_weights(self.conv.conv.weight, None, self.conv.bn)
        w = w + F.pad(self.conv1.weight, [p, p, p, p])
        identity = torch.zeros_like(w)
        identity[:, 0, p, p] = 1.0
        w = w + identity
        w, b = _fuse_conv_bn_weights(w, b, self.bn)
        self.reparam_conv = nn.Conv2d(self.channels, self.channels, k, 1, (k - 1) // 2,
                                      groups=self.channels, bias=True).to(w.device)
        self.reparam_conv.weight.copy_(w)
        self.reparam_conv.bias.copy_(b)
        del self.conv, self.conv1, self.bn
        self.is_repped = True


class LCNetV4Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_size, use_se=False, expand_ratio=2):
        super().__init__()
        self.has_residual = in_channels == out_channels and stride == 1
        token = OrderedDict()
        if self.has_residual:
            token["rep_dw"] = RepDWConv(in_channels, dw_size)
        else:
            token["dw_conv"] = Conv2D_BN(in_channels, in_channels, dw_size, stride,
                                         (dw_size - 1) // 2, groups=in_channels)
        if use_se:
            token["se"] = SELayer(in_channels)
        self.token_mixer = nn.Sequential(token)

        hidden = int(in_channels * expand_ratio)
        self.channel_mixer = nn.Sequential(OrderedDict([
            ("expand", Conv2D_BN(in_channels, hidden, 1, 1, 0)),
            ("act", nn.GELU()),
            ("compress", Conv2D_BN(hidden, out_channels, 1, 1, 0)),
        ]))

    def forward(self, x):
        x = self.token_mixer(x)
        if self.has_residual:
            return x + self.channel_mixer(x)
        return self.channel_mixer(x)


class PPLCNetV4(nn.Module):
    """PPLCNetV4 det backbone; returns [s1, s2, s3, s4] at strides 4/8/16/32."""

    def __init__(self, det=True, model_size="small", in_channels=3, **kwargs):
        super().__init__()
        if not det:
            raise NotImplementedError("Only the PPLCNetV4 det branch is ported")
        assert model_size in NET_CONFIG_DET, \
            f"det model_size must be one of {list(NET_CONFIG_DET)}, got '{model_size}'"
        cfg = NET_CONFIG_DET[model_size]
        stem_mid, stem_out = cfg["stem"]
        self.stem = StemBlock(in_channels, stem_mid, stem_out)

        def make_stage(key):
            return nn.Sequential(*[
                LCNetV4Block(in_c, out_c, s, k, se, expand_ratio=2)
                for k, in_c, out_c, s, se in cfg[key]
            ])

        self.blocks_s1 = make_stage("blocks_s1")
        self.blocks_s2 = make_stage("blocks_s2")
        self.blocks_s3 = make_stage("blocks_s3")
        self.blocks_s4 = make_stage("blocks_s4")
        self.out_channels = [cfg[k][-1][2] for k in ("blocks_s1", "blocks_s2", "blocks_s3", "blocks_s4")]

    def forward(self, x):
        x = self.stem(x)
        o1 = self.blocks_s1(x)
        o2 = self.blocks_s2(o1)
        o3 = self.blocks_s3(o2)
        o4 = self.blocks_s4(o3)
        return [o1, o2, o3, o4]
