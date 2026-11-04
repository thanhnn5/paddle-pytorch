import torch
import torch.nn as nn
import torch.nn.functional as F
from torchocr.modeling.common import Activation
from torchocr.modeling.backbones.det_mobilenet_v3 import SEModule
from torchocr.modeling.necks.intracl import IntraCLBlock

class DSConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride=1,
                 groups=None,
                 act="relu",
                 **kwargs):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.act = act
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias=False)

        self.bn2 = nn.BatchNorm2d(int(in_channels * 4))

        self.conv3 = nn.Conv2d(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False)
        self.act = None
        if act:
            self.act = Activation(act)
    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.act:
            x = self.act(x)

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class DBFPN(nn.Module):
    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf

        self.in2_conv = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.in3_conv = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.in4_conv = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.in5_conv = nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.p5_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.p4_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.p3_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.p2_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)

        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.interpolate(
            in5, scale_factor=2, mode="nearest", )#align_mode=1)  # 1/16
        out3 = in3 + F.interpolate(
            out4, scale_factor=2, mode="nearest", )#align_mode=1)  # 1/8
        out2 = in2 + F.interpolate(
            out3, scale_factor=2, mode="nearest", )#align_mode=1)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.interpolate(p5, scale_factor=8, mode="nearest", )#align_mode=1)
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest", )#align_mode=1)
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest", )#align_mode=1)

        fuse = torch.cat([p5, p4, p3, p2], dim=1)

        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        return fuse


class RSELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True, hardsigmoid_slope=None):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False)
        self.se_block = SEModule(self.out_channels, hardsigmoid_slope=hardsigmoid_slope)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, hardsigmoid_slope=None, **kwargs):
        """``hardsigmoid_slope``: SE gate slope, see ``SEModule``. Leave None
        (torch's 1/6) for models trained in torchocr; set 0.2 to match
        PaddleOCR (required for weights converted from Paddle)."""
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut,
                    hardsigmoid_slope=hardsigmoid_slope))
            self.inp_conv.append(
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut,
                    hardsigmoid_slope=hardsigmoid_slope))

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest")  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest")
        p4 = F.upsample(p4, scale_factor=4, mode="nearest")
        p3 = F.upsample(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class LKPAN(nn.Module):
    def __init__(self, in_channels, out_channels, mode='large', **kwargs):
        super(LKPAN, self).__init__()
        self.out_channels = out_channels

        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        # pan head
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()

        if mode.lower() == 'lite':
            p_layer = DSConv
        elif mode.lower() == 'large':
            p_layer = nn.Conv2d
        else:
            raise ValueError(
                "mode can only be one of ['lite', 'large'], but received {}".
                format(mode))

        for i in range(len(in_channels)):
            self.ins_conv.append(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    bias=False))

            self.inp_conv.append(
                p_layer(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    bias=False))

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2d(
                        in_channels=self.out_channels // 4,
                        out_channels=self.out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False))
            self.pan_lat_conv.append(
                p_layer(
                    in_channels=self.out_channels // 4,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    bias=False))
        self.intracl = False
        if 'intracl' in kwargs.keys() and kwargs['intracl'] is True:
            self.intracl = kwargs['intracl']
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest")  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest")
        p4 = F.upsample(p4, scale_factor=4, mode="nearest")
        p3 = F.upsample(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class ASFBlock(nn.Module):
    """
    This code is refered from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)

        self.spatial_scale = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                bias=False,
                padding=1,
                ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                bias=False,
                ),
            nn.Sigmoid())

        self.channel_scale = nn.Sequential(
            nn.Conv2d(
                in_channels=inter_channels,
                out_channels=out_features_num,
                kernel_size=1,
                bias=False,
                ),
            nn.Sigmoid())

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = torch.mean(fuse_features, dim=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i:i + 1] * features_list[i])
        return torch.cat(out_list, dim=1)


# ---------------------------------------------------------------------------
# PP-OCRv6 RepLKFPN (ported from PaddleOCR ppocr/modeling/necks/db_fpn.py).
# Parameter names mirror Paddle so weights convert by BN-buffer rename only.
# ---------------------------------------------------------------------------
class PaddleSEModule(nn.Module):
    """SE block matching ppocr.modeling.backbones.det_mobilenet_v3.SEModule.

    Paddle uses ``hardsigmoid(slope=0.2, offset=0.5)`` here, NOT torch's
    ``nn.Hardsigmoid`` (slope 1/6) used by torchocr's ``SEModule``.
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, inputs):
        x = inputs.mean(dim=(2, 3), keepdim=True)
        x = self.conv2(F.relu(self.conv1(x)))
        x = torch.clamp(0.2 * x + 0.5, 0.0, 1.0)
        return inputs * x


class PaddleRSELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.se_block = PaddleSEModule(out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        return x + self.se_block(x) if self.shortcut else self.se_block(x)


class DilatedReparamBlock(nn.Module):
    """UniRepLKNet Dilated Reparam Block (depthwise).

    Train: large-kernel DW conv + parallel dilated DW convs, each with BN.
    ``rep()`` merges every branch into one large-kernel DW conv (with bias).
    """

    _BRANCHES = {
        5: ([3, 3], [1, 2]),
        7: ([5, 3, 3], [1, 2, 3]),
        9: ([5, 5, 3, 3], [1, 2, 3, 4]),
        11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
        13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
    }

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        if kernel_size not in self._BRANCHES:
            raise ValueError(f"DilatedReparamBlock kernel_size must be in "
                             f"{sorted(self._BRANCHES)}, got {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.kernel_sizes, self.dilates = self._BRANCHES[kernel_size]
        self.is_repped = False

        self.lk_origin = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2,
                                   groups=channels, bias=False)
        self.origin_bn = nn.BatchNorm2d(channels)
        for k, r in zip(self.kernel_sizes, self.dilates):
            equiv_ks = r * (k - 1) + 1
            setattr(self, f"dil_conv_k{k}_{r}",
                    nn.Conv2d(channels, channels, k, 1, equiv_ks // 2, dilation=r,
                              groups=channels, bias=False))
            setattr(self, f"dil_bn_k{k}_{r}", nn.BatchNorm2d(channels))

    def forward(self, x):
        if self.is_repped:
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            out = out + getattr(self, f"dil_bn_k{k}_{r}")(getattr(self, f"dil_conv_k{k}_{r}")(x))
        return out

    @staticmethod
    def _fuse_bn(conv, bn):
        std = torch.sqrt(bn.running_var + bn.eps)
        return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * bn.weight / std

    @staticmethod
    def _dilate_kernel(kernel, r):
        """Insert (r-1) zeros between taps: (C,1,k,k) -> (C,1,r*(k-1)+1, ...)."""
        if r == 1:
            return kernel
        c, _, k, _ = kernel.shape
        out = kernel.new_zeros(c, 1, r * (k - 1) + 1, r * (k - 1) + 1)
        out[:, :, ::r, ::r] = kernel
        return out

    @torch.no_grad()
    def rep(self):
        if self.is_repped:
            return
        w, b = self._fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            bw, bb = self._fuse_bn(getattr(self, f"dil_conv_k{k}_{r}"), getattr(self, f"dil_bn_k{k}_{r}"))
            bw = self._dilate_kernel(bw, r)
            p = self.kernel_size // 2 - bw.shape[2] // 2
            w = w + (F.pad(bw, [p, p, p, p]) if p > 0 else bw)
            b = b + bb
        merged = nn.Conv2d(self.channels, self.channels, self.kernel_size, 1, self.kernel_size // 2,
                           groups=self.channels, bias=True).to(w.device)
        merged.weight.copy_(w)
        merged.bias.copy_(b)
        self.lk_origin = merged
        del self.origin_bn
        for k, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, f"dil_conv_k{k}_{r}")
            delattr(self, f"dil_bn_k{k}_{r}")
        self.is_repped = True


class RepLKFPN(nn.Module):
    """RSEFPN with inp_conv = DilatedReparamBlock(DW) + PW 1x1 + SE (PP-OCRv6 det).

    Eval returns the fused [B, out_channels, H/4, W/4] map; train additionally
    returns the 1/16, 1/8, 1/4 lateral maps for DBHead's auxiliary heads.
    """

    def __init__(self, in_channels, out_channels, shortcut=True, dilated_kernel_size=7, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.shortcut = shortcut
        self.ins_conv = nn.ModuleList()
        self.inp_conv_dw = nn.ModuleList()
        self.inp_conv_pw = nn.ModuleList()
        self.inp_conv_se = nn.ModuleList()
        for c in in_channels:
            self.ins_conv.append(PaddleRSELayer(c, out_channels, kernel_size=1, shortcut=shortcut))
            self.inp_conv_dw.append(DilatedReparamBlock(out_channels, dilated_kernel_size))
            self.inp_conv_pw.append(nn.Conv2d(out_channels, out_channels // 4, 1, bias=False))
            self.inp_conv_se.append(PaddleSEModule(out_channels // 4))

    def _inp_forward(self, x, idx):
        x = self.inp_conv_pw[idx](self.inp_conv_dw[idx](x))
        se = self.inp_conv_se[idx](x)
        return x + se if self.shortcut else se

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")   # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        p5 = F.interpolate(self._inp_forward(in5, 3), scale_factor=8, mode="nearest")
        p4 = F.interpolate(self._inp_forward(out4, 2), scale_factor=4, mode="nearest")
        p3 = F.interpolate(self._inp_forward(out3, 1), scale_factor=2, mode="nearest")
        p2 = self._inp_forward(out2, 0)

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse


class DilatedReparamConv(nn.Module):
    """DilatedReparamBlock(DW, in_ch) -> 1x1 PW conv (in_ch -> out_ch) -> BN.

    Drop-in for a large-kernel standard conv (PaddleOCR db_fpn.DilatedReparamConv).
    ``rep()`` merges the DW branches and fuses PW+BN into one biased 1x1 conv.
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, **kwargs):
        super().__init__()
        self.is_repped = False
        self.dw = DilatedReparamBlock(in_channels, kernel_size)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pw(self.dw(x))
        if not self.is_repped:
            x = self.bn(x)
        return x

    @torch.no_grad()
    def rep(self):
        if self.is_repped:
            return
        self.dw.rep()
        w, b = DilatedReparamBlock._fuse_bn(self.pw, self.bn)
        fused = nn.Conv2d(self.pw.in_channels, self.pw.out_channels, 1, bias=True).to(w.device)
        fused.weight.copy_(w)
        fused.bias.copy_(b)
        self.pw = fused
        del self.bn
        self.is_repped = True


class RepLKPAN(nn.Module):
    """LKPAN with every 9x9 standard conv (4 inp_conv + 4 pan_lat_conv) replaced
    by DilatedReparamConv (PP-OCRv6 medium det; PaddleOCR db_fpn.RepLKPAN).

    Eval returns the fused [B, out_channels, H/4, W/4] map; train additionally
    returns the 1/16, 1/8, 1/4 lateral maps for DBHead's auxiliary heads.
    """

    def __init__(self, in_channels, out_channels, mode="large", **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.is_repped = False
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()
        for i, c in enumerate(in_channels):
            self.ins_conv.append(nn.Conv2d(c, out_channels, 1, bias=False))
            self.inp_conv.append(DilatedReparamConv(out_channels, out_channels // 4, kernel_size=9))
            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2d(out_channels // 4, out_channels // 4, 3, stride=2, padding=1, bias=False))
            self.pan_lat_conv.append(DilatedReparamConv(out_channels // 4, out_channels // 4, kernel_size=9))

        self.intracl = kwargs.get('intracl', False) is True
        if self.intracl:
            self.incl1 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(out_channels // 4, reduce_factor=2)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")   # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse

    def rep(self):
        if self.is_repped:
            return
        for m in list(self.inp_conv) + list(self.pan_lat_conv):
            m.rep()
        self.is_repped = True
