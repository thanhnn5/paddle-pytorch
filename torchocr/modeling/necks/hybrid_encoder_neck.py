"""
HybridEncoderNeck: wraps DEIMv2's HybridEncoder for use in PytorchOCR's BaseModel pipeline.

Architecture:
    (c2, c3, c4) @ strides [8, 16, 32]
        -> HybridEncoder  -> [p2, p3, p4] @ strides [8, 16, 32]
        -> merge + 2x upsample             -> [B, hidden_dim, H/4, W/4]
        -> merge_conv                      -> [B, out_channels, H/4, W/4]
"""

import importlib.util
import os
import sys
import types

import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make DEIMv2 importable without a setup.py.
# We load engine.deim.hybrid_encoder directly via importlib stubs to avoid
# importing the full engine package (which requires optional deps like
# calflops, yaml_config, etc.).
# ---------------------------------------------------------------------------
_deimv2_path = os.environ.get(
    "DEIMV2_PATH",
    os.path.join(os.path.dirname(__file__), "../../../../DEIMv2"),
)
_deimv2_path = os.path.abspath(_deimv2_path)
if _deimv2_path not in sys.path:
    sys.path.insert(0, _deimv2_path)


def _load_hybrid_encoder():
    """
    Load engine.deim.hybrid_encoder without triggering engine/__init__.py,
    which pulls in heavy/optional dependencies (calflops, yaml_config, etc.).
    """
    # Register stub packages so relative imports inside the modules resolve.
    for pkg, subpath in [
        ("engine", "engine"),
        ("engine.deim", os.path.join("engine", "deim")),
        ("engine.core", os.path.join("engine", "core")),
        ("engine.backbone", os.path.join("engine", "backbone")),
    ]:
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = [os.path.join(_deimv2_path, subpath)]
            stub.__package__ = pkg
            sys.modules[pkg] = stub

    # Load engine.core so @register() decorator works
    core_dir = os.path.join(_deimv2_path, "engine", "core")
    if "engine.core" not in sys.modules or not hasattr(sys.modules["engine.core"], "register"):
        spec = importlib.util.spec_from_file_location(
            "engine.core",
            os.path.join(core_dir, "__init__.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "engine.core"
        mod.__path__ = [core_dir]
        sys.modules["engine.core"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[HybridEncoderNeck] WARNING: required module load failed: {e} — HybridEncoder may not initialize correctly")

    # Expose register on the engine stub so `from ..core import register` works
    engine_stub = sys.modules["engine"]
    if not hasattr(engine_stub, "core"):
        engine_stub.core = sys.modules["engine.core"]

    # Load engine.deim.utils (provides get_activation used by hybrid_encoder)
    deim_dir = os.path.join(_deimv2_path, "engine", "deim")
    if "engine.deim.utils" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "engine.deim.utils",
            os.path.join(deim_dir, "utils.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "engine.deim"
        sys.modules["engine.deim.utils"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[HybridEncoderNeck] WARNING: required module load failed: {e} — HybridEncoder may not initialize correctly")

    # Expose utils on the engine.deim stub
    deim_stub = sys.modules["engine.deim"]
    if not hasattr(deim_stub, "utils"):
        deim_stub.utils = sys.modules.get("engine.deim.utils")

    # Load engine.deim.hybrid_encoder
    spec = importlib.util.spec_from_file_location(
        "engine.deim.hybrid_encoder",
        os.path.join(deim_dir, "hybrid_encoder.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "engine.deim"
    sys.modules["engine.deim.hybrid_encoder"] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    _hybrid_encoder_mod = _load_hybrid_encoder()
except Exception as e:
    raise ImportError(
        f"HybridEncoderNeck failed to load DEIMv2's HybridEncoder: {e}\n"
        f"Set DEIMV2_PATH env var to the DEIMv2 repo root."
    ) from e
HybridEncoder = _hybrid_encoder_mod.HybridEncoder


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
