"""Stage-2 feature distillation: ConvNeXt-V2 femto student <- DINOv3-ConvNeXt-tiny teacher.

Reads a folder of unlabeled images, runs both teacher (frozen) and student over
identical augmented crops, and minimizes per-channel-normalized MSE on the last
two stage outputs (strides 16 and 32). Per docs/convnext_distillation.md.

Example:
    python tools/distill/train.py \\
        --images /Users/thanhnn5/Downloads/pod \\
        --teacher-ckpt weights/dinov3/convnext_det_unfreeze.pth \\
        --output output/distill_convnext_femto \\
        --device auto --epochs 100 --batch-size 64 --img-size 640
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchocr.modeling.backbones.det_convnext import ConvNeXtDetBackbone
from torchocr.modeling.backbones.det_convnextv2 import ConvNeXtV2Backbone
from torchocr.utils.ckpt import load_pretrained_params
from torchocr.utils.logging import get_logger
from tools.distill.dataset import UnlabeledImageFolder, build_train_transform


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_for(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        # MPS bf16 is supported on torch >= 2.3 but flaky for some ops; use fp16.
        return torch.amp.autocast("mps", dtype=torch.float16)
    return nullcontext()


class StageAdapter(nn.Module):
    """1x1 conv mapping student stage channels -> teacher stage channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        return self.proj(x)


def channel_layernorm(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-token channel-wise LN: normalize each spatial position over channels."""
    mean = feat.mean(dim=1, keepdim=True)
    var = feat.var(dim=1, keepdim=True, unbiased=False)
    return (feat - mean) / torch.sqrt(var + eps)


def cosine_sim_per_stage(student_feat, teacher_feat):
    s = F.normalize(student_feat.flatten(2), dim=1)
    t = F.normalize(teacher_feat.flatten(2), dim=1)
    return (s * t).sum(dim=1).mean().item()


def build_teacher(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load the task-adapted DINOv3-ConvNeXt-tiny backbone from a BaseModel ckpt."""
    teacher = ConvNeXtDetBackbone(pretrained=True, finetune=False)
    # The checkpoint is a BaseModel state_dict: keys prefixed `backbone.`, `neck.`, `head.`.
    # Strip the prefix for the keys we want.
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    backbone_state = {}
    for k, v in raw.items():
        if k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
    missing, unexpected = teacher.load_state_dict(backbone_state, strict=False)
    logger = get_logger()
    logger.info(f"teacher load: matched {len(backbone_state)} keys, "
                f"missing={len(missing)}, unexpected={len(unexpected)}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.to(device)


def build_student(size: str, device: torch.device) -> ConvNeXtV2Backbone:
    student = ConvNeXtV2Backbone(size=size, pretrained=True, finetune=True)
    return student.to(device)


def build_adapters(student_ch, teacher_ch, align_stages, device):
    """Build one adapter per stage we align."""
    return nn.ModuleList([
        StageAdapter(student_ch[s], teacher_ch[s]) for s in align_stages
    ]).to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="root folder of unlabeled POD images")
    p.add_argument("--teacher-ckpt", required=True,
                   help="path to BaseModel checkpoint with ConvNeXtDetBackbone weights")
    p.add_argument("--output", default="output/distill_convnext_femto")
    p.add_argument("--student-size", default="femto",
                   choices=["atto", "femto", "pico", "nano"])
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0,
                   help="cap total steps (0 = unlimited; useful for smoke tests)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--align-stages", type=int, nargs="+", default=[2, 3],
                   help="0-indexed stage indices to align (last two by default)")
    p.add_argument("--cosine-weight", type=float, default=0.5,
                   help="weight for the cosine-similarity loss term")
    p.add_argument("--ema-decay", type=float, default=0.9995,
                   help="EMA decay; set to 0 to disable. Use 0.999 for short runs "
                        "(<50k steps), 0.9999 for very long runs (>500k steps).")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp", action="store_true", help="enable mixed-precision autocast")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=5, help="epochs between checkpoints")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def warmup_cosine_lr(step, total_steps, warmup_steps, peak_lr, min_lr=1e-6):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    logger = get_logger()
    device = pick_device(args.device)
    logger.info(f"device: {device}")

    teacher = build_teacher(args.teacher_ckpt, device)
    student = build_student(args.student_size, device)

    teacher_ch = teacher.out_channels      # [96, 192, 384, 768]
    student_ch = student.out_channels      # [48, 96, 192, 384] for femto
    logger.info(f"teacher out_channels: {teacher_ch}")
    logger.info(f"student out_channels: {student_ch}")
    align = sorted(set(args.align_stages))
    adapters = build_adapters(student_ch, teacher_ch, align, device)
    logger.info(f"aligning stages (0-indexed): {align}")

    transform = build_train_transform(img_size=args.img_size)
    dataset = UnlabeledImageFolder(args.images, transform)
    logger.info(f"dataset: {len(dataset)} images at {args.images}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0), drop_last=True,
    )

    params = [
        {"params": student.parameters()},
        {"params": adapters.parameters()},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))

    steps_per_epoch = max(1, len(loader))
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    logger.info(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    ema_state = None
    if args.ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in student.state_dict().items()}

    history = []
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            if args.max_steps and global_step >= args.max_steps:
                break

            lr = warmup_cosine_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x = batch.to(device, non_blocking=True)
            with torch.no_grad(), autocast_for(device, args.amp):
                t_feats = teacher(x)

            with autocast_for(device, args.amp):
                s_feats = student(x)
                mse_per_stage, cos_per_stage = [], []
                for adapter, stage_idx in zip(adapters, align):
                    s = adapter(s_feats[stage_idx])
                    t = t_feats[stage_idx]
                    s_n = channel_layernorm(s)
                    t_n = channel_layernorm(t)
                    mse = F.mse_loss(s_n, t_n)
                    mse_per_stage.append(mse)
                    cos_per_stage.append(cosine_sim_per_stage(s.detach().float(),
                                                              t.detach().float()))
                loss_mse = sum(mse_per_stage) / len(mse_per_stage)
                loss_cos = sum((1.0 - c) for c in cos_per_stage) / len(cos_per_stage)
                loss = loss_mse + args.cosine_weight * loss_cos

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                nn.utils.clip_grad_norm_(adapters.parameters(), args.grad_clip)
            optimizer.step()

            if ema_state is not None:
                sd = student.state_dict()
                d = args.ema_decay
                for k in ema_state:
                    if ema_state[k].dtype.is_floating_point:
                        ema_state[k].mul_(d).add_(sd[k].detach(), alpha=1 - d)
                    else:
                        ema_state[k].copy_(sd[k])

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                msg = (f"ep={epoch} step={global_step}/{total_steps} "
                       f"lr={lr:.2e} loss={loss.item():.4f} "
                       f"mse=[{','.join(f'{m.item():.4f}' for m in mse_per_stage)}] "
                       f"cos=[{','.join(f'{c:.3f}' for c in cos_per_stage)}] "
                       f"elapsed={elapsed:.0f}s")
                logger.info(msg)
                history.append({
                    "epoch": epoch, "step": global_step, "lr": lr,
                    "loss": float(loss.item()),
                    "mse": [float(m.item()) for m in mse_per_stage],
                    "cos": [float(c) for c in cos_per_stage],
                    "elapsed_s": elapsed,
                })

            global_step += 1
        else:
            # save end-of-epoch checkpoint
            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                ckpt = {
                    "epoch": epoch + 1,
                    "student": student.state_dict(),
                    "adapters": adapters.state_dict(),
                    "ema": ema_state,
                    "args": vars(args),
                }
                path = os.path.join(args.output, f"student_ep{epoch+1}.pth")
                torch.save(ckpt, path)
                logger.info(f"saved {path}")
            continue
        # broke out of inner loop because of max-steps
        break

    final = os.path.join(args.output, "student_final.pth")
    torch.save({
        "student": student.state_dict(),
        "adapters": adapters.state_dict(),
        "ema": ema_state,
        "args": vars(args),
    }, final)
    with open(os.path.join(args.output, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"done. final={final}")


if __name__ == "__main__":
    main()
