"""SE hardsigmoid slope option (SEModule / RSEFPN).

PaddleOCR's det SEModule gates with hardsigmoid(slope=0.2, offset=0.5);
torch's nn.Hardsigmoid is slope 1/6. Weights converted from Paddle need 0.2
(PP-OCRv5 mobile det: 17/94 boxes change at 1/6); torch-trained models keep
the default.
"""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchocr.modeling.backbones.det_mobilenet_v3 import SEModule  # noqa: E402
from torchocr.modeling.common import HardSigmoid  # noqa: E402
from torchocr.modeling.necks.db_fpn import RSEFPN  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _se_gates(module):
    return [m.hard_sigmoid for m in module.modules() if isinstance(m, SEModule)]


def test_hardsigmoid_formula():
    x = torch.linspace(-10, 10, 101)
    torch.testing.assert_close(HardSigmoid(0.2)(x), torch.clamp(0.2 * x + 0.5, 0, 1))
    # slope 1/6 reproduces torch's nn.Hardsigmoid
    torch.testing.assert_close(HardSigmoid(1 / 6)(x), nn.Hardsigmoid()(x))


def test_semodule_default_is_torch_hardsigmoid():
    se = SEModule(16)
    assert isinstance(se.hard_sigmoid.act, nn.Hardsigmoid)


def test_semodule_slope_changes_output_not_params():
    torch.manual_seed(0)
    default, paddle = SEModule(16), SEModule(16, hardsigmoid_slope=0.2)
    assert default.state_dict().keys() == paddle.state_dict().keys()
    paddle.load_state_dict(default.state_dict())
    x = torch.randn(2, 16, 8, 8) * 5
    with torch.no_grad():
        s = default.conv2(default.relu1(default.conv1(default.avg_pool(x))))
        torch.testing.assert_close(paddle(x), x * torch.clamp(0.2 * s + 0.5, 0, 1))
        assert not torch.allclose(default(x), paddle(x))


def test_rsefpn_threads_slope_to_every_se_block():
    in_ch = [16, 24, 56, 480]
    default = RSEFPN(in_ch, 96)
    paddle = RSEFPN(in_ch, 96, hardsigmoid_slope=0.2)
    assert len(_se_gates(paddle)) == 8
    assert all(isinstance(g, HardSigmoid) and g.slope == 0.2 for g in _se_gates(paddle))
    assert not any(isinstance(g, HardSigmoid) for g in _se_gates(default))
    # same parameters, so checkpoints load into either variant
    assert default.state_dict().keys() == paddle.state_dict().keys()


def test_ppocrv5_mobile_det_config_uses_paddle_slope():
    from torchocr import Config
    from torchocr.modeling.architectures import build_model

    cfg = Config(os.path.join(REPO, "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml")).cfg
    assert cfg["Architecture"]["Neck"]["hardsigmoid_slope"] == 0.2
    model = build_model(cfg["Architecture"])
    gates = _se_gates(model.neck)
    assert len(gates) == 8 and all(isinstance(g, HardSigmoid) and g.slope == 0.2 for g in gates)
