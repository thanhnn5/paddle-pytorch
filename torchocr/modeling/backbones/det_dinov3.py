"""
DINOv3DetBackbone: wraps DEIMv2's DINOv3STAs for use in PytorchOCR's BaseModel pipeline.

Architecture:
    Image -> DINOv3DetBackbone -> HybridEncoderNeck -> PFHeadLocal (DBHead)
"""

import importlib.util
import os
import sys
import types

import torch.nn as nn

# ---------------------------------------------------------------------------
# Make DEIMv2 importable without a setup.py.
# We load dinov3_adapter directly from its file to avoid importing the full
# engine package (which requires optional deps like calflops).
# ---------------------------------------------------------------------------
_deimv2_path = os.environ.get(
    "DEIMV2_PATH",
    os.path.join(os.path.dirname(__file__), "../../../../DEIMv2"),
)
_deimv2_path = os.path.abspath(_deimv2_path)
if _deimv2_path not in sys.path:
    sys.path.insert(0, _deimv2_path)


def _load_dinov3_adapter():
    """
    Load engine.backbone.dinov3_adapter without triggering engine/__init__.py,
    which pulls in heavy/optional dependencies (calflops, etc.).
    """
    backbone_dir = os.path.join(_deimv2_path, "engine", "backbone")

    # Register stub packages so relative imports inside the module resolve.
    for pkg in ("engine", "engine.backbone"):
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = [
                os.path.join(_deimv2_path, *pkg.split("."))
            ]
            stub.__package__ = pkg
            sys.modules[pkg] = stub

    # Load engine.backbone.common first (provides get_activation etc.)
    for mod_name, filename in [
        ("engine.backbone.common", "common.py"),
        ("engine.backbone.utils", "utils.py"),
        ("engine.backbone.vit_tiny", "vit_tiny.py"),
    ]:
        if mod_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                mod_name, os.path.join(backbone_dir, filename)
            )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "engine.backbone"
            sys.modules[mod_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass  # tolerate optional sub-deps in these helpers

    # Load engine.core so @register() decorator works
    core_dir = os.path.join(_deimv2_path, "engine", "core")
    for mod_name, filename in [
        ("engine.core", "__init__.py"),
    ]:
        if mod_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                mod_name, os.path.join(core_dir, filename)
            )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "engine.core"
            mod.__path__ = [core_dir]
            sys.modules[mod_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass

    # Load the dinov3 sub-package __init__
    dinov3_dir = os.path.join(backbone_dir, "dinov3")
    if "engine.backbone.dinov3" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "engine.backbone.dinov3",
            os.path.join(dinov3_dir, "__init__.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "engine.backbone.dinov3"
        mod.__path__ = [dinov3_dir]
        sys.modules["engine.backbone.dinov3"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass

    # Finally load the adapter itself
    adapter_path = os.path.join(backbone_dir, "dinov3_adapter.py")
    spec = importlib.util.spec_from_file_location(
        "engine.backbone.dinov3_adapter", adapter_path
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "engine.backbone"
    sys.modules["engine.backbone.dinov3_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod


_adapter_mod = _load_dinov3_adapter()
DINOv3STAs = _adapter_mod.DINOv3STAs


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
                bn.weight = child.weight
                bn.bias = child.bias
            if child.track_running_stats:
                bn.running_mean = child.running_mean
                bn.running_var = child.running_var
                bn.num_batches_tracked = child.num_batches_tracked
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
        interaction_indexes: list = [5, 8, 11],
        hidden_dim: int = 256,
        conv_inplane: int = 32,
        use_sta: bool = True,
        finetune: bool = True,
        sync_bn: bool = False,              # True only under torch DDP
    ):
        super().__init__()

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

        # PytorchOCR BaseModel reads this to set Neck's in_channels
        self.out_channels = [hidden_dim, hidden_dim, hidden_dim]

    def forward(self, x):
        """Return (c2, c3, c4) feature maps at strides 8, 16, 32."""
        return self._backbone(x)
