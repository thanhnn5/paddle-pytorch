"""
Quick sanity check: build model from config, run forward + backward, print summary.

Usage:
    python tools/verify_trainable.py -c configs/det/PP-OCRv5/PP-OCRv5_dinov3_det.yml
"""

import argparse
import sys
import os
import yaml
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchocr.modeling.architectures.base_model import BaseModel


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    arch_cfg = cfg["Architecture"]
    H = W = args.img_size
    B = args.batch_size

    # --- Build model ---
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Input:  [{B}, 3, {H}, {W}]")
    print("=" * 60)

    model = BaseModel(arch_cfg)
    model.train()

    total, trainable, frozen = count_params(model)
    print(f"\nParameters:")
    print(f"  Total:     {total:>12,}")
    print(f"  Trainable: {trainable:>12,}")
    print(f"  Frozen:    {frozen:>12,}")

    # --- Per-component breakdown ---
    print(f"\nComponent breakdown:")
    for name, child in model.named_children():
        t, tr, fr = count_params(child)
        print(f"  {name:20s}  total={t:>10,}  trainable={tr:>10,}  frozen={fr:>10,}")

    # --- Forward pass ---
    print(f"\nForward pass...")
    x = torch.randn(B, 3, H, W)
    try:
        out = model(x)
        if isinstance(out, dict):
            for k, v in out.items():
                shape = v.shape if isinstance(v, torch.Tensor) else type(v)
                print(f"  output['{k}']: {shape}")
        elif isinstance(out, (list, tuple)):
            for i, v in enumerate(out):
                shape = v.shape if isinstance(v, torch.Tensor) else type(v)
                print(f"  output[{i}]: {shape}")
        elif isinstance(out, torch.Tensor):
            print(f"  output: {out.shape}")
        else:
            print(f"  output type: {type(out)}")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # --- Backward pass ---
    print(f"\nBackward pass...")
    try:
        if isinstance(out, dict):
            loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor))
        elif isinstance(out, (list, tuple)):
            loss = sum(v.mean() for v in out if isinstance(v, torch.Tensor))
        else:
            loss = out.mean()

        loss.backward()
        print(f"  loss = {loss.item():.6f}")

        # Check gradients
        has_grad = 0
        no_grad = 0
        nan_grad = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        nan_grad += 1
                        print(f"  WARNING: NaN gradient in {name}")
                    else:
                        has_grad += 1
                else:
                    no_grad += 1

        print(f"\n  Params with gradients:    {has_grad}")
        print(f"  Params without gradients: {no_grad}")
        if nan_grad > 0:
            print(f"  Params with NaN grads:    {nan_grad}")

    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # --- Summary ---
    print("\n" + "=" * 60)
    if nan_grad > 0:
        print("RESULT: WARN - model runs but has NaN gradients")
    else:
        print("RESULT: OK - model is trainable")
    print("=" * 60)


if __name__ == "__main__":
    main()
