"""Convert a Stage-2 distillation checkpoint into a BaseModel-compatible
pretrained-weights file for Stage-3 detection fine-tune.

The distillation checkpoint has the shape:
    {
        "student":  state_dict of the femto backbone (no prefix),
        "adapters": 1x1 conv adapter weights (Stage-2 only; discarded here),
        "ema":      EMA backbone weights (same key layout as `student`),
        "args":     CLI args used,
    }

`tools/train.py` -> `load_pretrained_params` expects a flat state_dict whose
keys match the BaseModel state_dict (i.e. prefixed `backbone.<...>`,
`neck.<...>`, `head.<...>`). We emit just the backbone keys, prefixed.

Example:
    python tools/distill/extract_backbone.py \\
        --ckpt output/distill_convnext_femto_long/student_final.pth \\
        --out  weights/distilled/femto_backbone.pth \\
        --use-ema
"""

import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Stage-2 distillation checkpoint")
    p.add_argument("--out", required=True, help="output .pth for Stage-3 init")
    p.add_argument("--use-ema", action="store_true",
                   help="prefer EMA weights over raw student weights "
                        "(recommended; usually 0.5-1.5 pts higher H-mean).")
    p.add_argument("--key", default="student",
                   help="which top-level key in the distill ckpt to read "
                        "(default 'student'; use 'ema' to read the EMA dict "
                        "directly).")
    args = p.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise SystemExit(f"unexpected ckpt type {type(raw).__name__}; "
                         "expected dict with 'student'/'ema' keys")

    if args.use_ema:
        if raw.get("ema") is None:
            raise SystemExit("--use-ema specified but ckpt has no 'ema' "
                             "weights (was --ema-decay 0 during distillation?)")
        sd = raw["ema"]
        source = "ema"
    else:
        if args.key not in raw:
            raise SystemExit(f"key '{args.key}' not found; available: "
                             f"{list(raw.keys())}")
        sd = raw[args.key]
        source = args.key

    # Prefix every key with `backbone.` so it loads into BaseModel.
    prefixed = {f"backbone.{k}": v for k, v in sd.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(prefixed, args.out)

    print(f"source       : {args.ckpt}  (using '{source}' weights)")
    print(f"output       : {args.out}")
    print(f"keys written : {len(prefixed)} (all prefixed with 'backbone.')")
    sample = list(prefixed.keys())[:3]
    print(f"sample keys  : {sample}")


if __name__ == "__main__":
    main()
