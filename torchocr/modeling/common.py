
import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)

class HardSigmoid(nn.Module):
    """``clamp(slope * x + offset, 0, 1)``.

    torch's ``nn.Hardsigmoid`` is fixed at slope 1/6. Paddle's
    ``F.hardsigmoid`` defaults to slope 0.1666667 too, but PaddleOCR's det
    ``SEModule`` (MobileNetV3 backbone, RSEFPN / RepLKFPN necks) calls it with
    ``slope=0.2, offset=0.5``; weights converted from Paddle need that slope.
    """

    def __init__(self, slope=0.2, offset=0.5):
        super(HardSigmoid, self).__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return torch.clamp(x * self.slope + self.offset, 0.0, 1.0)

    def extra_repr(self):
        return f"slope={self.slope}, offset={self.offset}"


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'hard_sigmoid':
            self.act = nn.Hardsigmoid(inplace)
        elif act_type == 'hard_swish':
            self.act = nn.Hardswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act = GELU(inplace=inplace)
        elif act_type == 'swish':
            self.act = Swish(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)