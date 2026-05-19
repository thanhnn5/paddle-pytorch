"""Stage-2 feature distillation: orchestrator.

Reads a folder of unlabeled images, runs both teacher (frozen) and student over
identical augmented crops, and minimizes per-channel-normalized MSE on the last
two stage outputs (strides 16 and 32). Per docs/convnext_distillation.md.

Modules:
    dataset.py   image folder + augmentation
    model.py     teacher / student / adapter builders
    loss.py      feature-distillation loss (saliency-weighting will land here)
    lr.py        learning-rate schedule
    probe.py     held-out feature-fidelity probe
    utils.py     seeding, device, autocast, feature ops

Example:
    python tools/distill/train.py \\
        --images /Users/thanhnn5/Downloads/pod \\
        --teacher-ckpt weights/dinov3/convnext_det_unfreeze.pth \\
        --output output/distill_convnext_femto \\
        --device auto --epochs 100 --batch-size 64 --img-size 384
"""

import argparse
import json
import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchocr.utils.logging import get_logger
from tools.distill.dataset import UnlabeledImageFolder, build_train_transform
from tools.distill.loss import distill_loss
from tools.distill.lr import warmup_cosine_lr
from tools.distill.model import build_adapters, build_student, build_teacher
from tools.distill.probe import build_probe_batches, run_probe
from tools.distill.utils import (
    autocast_for, enable_deterministic_mode, pick_device,
    seed_everything, seed_worker,
)


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
    p.add_argument("--no-layernorm", action="store_true",
                   help="disable per-stage channel-wise LN before MSE")
    # Saliency-weighted loss (Option B; see docs/distill_saliency_weighted_loss.md).
    p.add_argument("--saliency", action="store_true",
                   help="weight per-pixel loss by the teacher's DBHead "
                        "probability map. Requires --teacher-config.")
    p.add_argument("--teacher-config", default=None,
                   help="YAML config for the full BaseModel teacher; needed "
                        "for --saliency (so we can build the full architecture "
                        "and read the prob map from the head).")
    p.add_argument("--saliency-alpha", type=float, default=0.05,
                   help="background weight floor (>=0.05 recommended). "
                        "Empirical default per docs/distill_saliency_weighted_loss.md.")
    p.add_argument("--saliency-beta", type=float, default=10.0,
                   help="foreground weight multiplier (text regions get "
                        "alpha + beta*p weight where p in [0,1]). High beta "
                        "because POD prob maps are sparse (mean ~0.03); see doc.")
    p.add_argument("--saliency-power", type=float, default=0.3,
                   help="raise saliency to this power before use; <1 softens "
                        "extreme confidence, >1 sharpens. Default 0.3 was "
                        "the clear winner on POD in a 600-step refine sweep "
                        "(sal_mean→0.34, stage-3 mse_ln ~10%% lower than p=0.5/0.7). "
                        "See docs/distill_saliency_weighted_loss.md.")
    p.add_argument("--ema-decay", type=float, default=0.9995,
                   help="EMA decay; set to 0 to disable. Use 0.999 for short runs "
                        "(<50k steps), 0.9999 for very long runs (>500k steps).")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp", action="store_true", help="enable mixed-precision autocast")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=5, help="epochs between checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="enable torch.use_deterministic_algorithms(True) + "
                        "cudnn deterministic mode. Required for bit-exact "
                        "reproducibility on CUDA; ~10-30%% slower. MPS still "
                        "has hardware-level atomic nondeterminism that this "
                        "flag cannot fix.")
    # Held-out probe
    p.add_argument("--probe-images", default=None,
                   help="folder of held-out images for clean cosine/MSE probes "
                        "(default: reuse --images with deterministic order)")
    p.add_argument("--probe-count", type=int, default=64,
                   help="number of probe images to use; 0 disables")
    p.add_argument("--probe-every", type=int, default=200, help="run probe every N steps")
    p.add_argument("--probe-batch-size", type=int, default=16)
    # W&B
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="convnext-distill")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    return p.parse_args()


def update_ema(ema_state: dict, model_state: dict, decay: float):
    for k in ema_state:
        if ema_state[k].dtype.is_floating_point:
            ema_state[k].mul_(decay).add_(model_state[k].detach(), alpha=1 - decay)
        else:
            ema_state[k].copy_(model_state[k])


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.deterministic:
        enable_deterministic_mode()
    os.makedirs(args.output, exist_ok=True)

    logger = get_logger()
    device = pick_device(args.device)
    logger.info(f"device: {device}")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or os.path.basename(args.output.rstrip("/")),
            entity=args.wandb_entity,
            tags=args.wandb_tags,
            config=vars(args),
            dir=args.output,
        )
        logger.info(f"wandb: {wandb_run.url}")

    if args.saliency and not args.teacher_config:
        raise SystemExit("--saliency requires --teacher-config (path to "
                         "the BaseModel YAML, e.g. "
                         "configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml)")
    teacher = build_teacher(
        args.teacher_ckpt, device,
        full_model_config=args.teacher_config if args.saliency else None,
    )
    if args.saliency:
        logger.info(f"saliency ON: alpha={args.saliency_alpha} "
                    f"beta={args.saliency_beta} power={args.saliency_power}")
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
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )

    optimizer = torch.optim.AdamW(
        [{"params": student.parameters()}, {"params": adapters.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    steps_per_epoch = max(1, len(loader))
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    logger.info(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
                f"warmup_steps={warmup_steps}")

    ema_state = None
    if args.ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in student.state_dict().items()}

    probe_batches = None
    if args.probe_count > 0:
        probe_root = args.probe_images or args.images
        probe_batches = build_probe_batches(
            probe_root, args.probe_count, args.img_size,
            args.probe_batch_size, args.num_workers,
        )
        logger.info(f"probe set: {sum(b.size(0) for b in probe_batches)} images "
                    f"from {probe_root}")

    history, probe_history = [], []
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
                t_feats, prob_map = teacher(x)

            saliency = None
            if args.saliency and prob_map is not None:
                saliency = prob_map.detach().float()
                if args.saliency_power != 1.0:
                    saliency = saliency.clamp_min(0).pow(args.saliency_power)

            with autocast_for(device, args.amp):
                s_feats = student(x)
                loss, per_stage, loss_mse, loss_cos = distill_loss(
                    s_feats, t_feats, adapters, align,
                    cosine_weight=args.cosine_weight,
                    layernorm=not args.no_layernorm,
                    saliency=saliency,
                    sal_alpha=args.saliency_alpha,
                    sal_beta=args.saliency_beta,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                nn.utils.clip_grad_norm_(adapters.parameters(), args.grad_clip)
            optimizer.step()

            if ema_state is not None:
                update_ema(ema_state, student.state_dict(), args.ema_decay)

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                mse_strs = ",".join(f"{ps.mse.item():.4f}" for ps in per_stage)
                cos_strs = ",".join(f"{ps.cos_sim:.3f}" for ps in per_stage)
                sal_str = ""
                if saliency is not None:
                    sal_mean = float(saliency.mean().item())
                    sal_p50 = float(saliency.flatten(1).median(dim=1).values.mean().item())
                    sal_frac = float((saliency > 0.5).float().mean().item())
                    sal_str = f" sal=mean{sal_mean:.3f}/p50{sal_p50:.3f}/frac>0.5={sal_frac:.3f}"
                logger.info(
                    f"ep={epoch} step={global_step}/{total_steps} "
                    f"lr={lr:.2e} loss={loss.item():.4f} "
                    f"mse=[{mse_strs}] cos=[{cos_strs}]{sal_str} "
                    f"elapsed={elapsed:.0f}s"
                )
                history.append({
                    "epoch": epoch, "step": global_step, "lr": lr,
                    "loss": float(loss.item()),
                    "mse": [float(ps.mse.item()) for ps in per_stage],
                    "cos": [float(ps.cos_sim) for ps in per_stage],
                    "elapsed_s": elapsed,
                })
                if wandb_run is not None:
                    log = {
                        "train/loss": float(loss.item()),
                        "train/loss_mse": float(loss_mse.item()),
                        "train/loss_cos": float(loss_cos),
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/elapsed_s": elapsed,
                    }
                    for i, st in enumerate(align):
                        log[f"train/mse_stage{st}"] = float(per_stage[i].mse.item())
                        log[f"train/cos_stage{st}"] = float(per_stage[i].cos_sim)
                    wandb_run.log(log, step=global_step)

            if probe_batches and global_step > 0 and global_step % args.probe_every == 0:
                probe = run_probe(teacher, student, adapters, align,
                                  probe_batches, device, args.amp)
                probe["step"] = global_step
                probe["epoch"] = epoch
                probe_history.append(probe)
                logger.info(
                    f"  PROBE step={global_step} "
                    f"cos=[{','.join(f'{c:.3f}' for c in probe['cos'])}] "
                    f"mse_ln=[{','.join(f'{m:.4f}' for m in probe['mse_ln'])}]"
                )
                if wandb_run is not None:
                    plog = {}
                    for i, st in enumerate(align):
                        plog[f"probe/cos_stage{st}"] = probe["cos"][i]
                        plog[f"probe/mse_ln_stage{st}"] = probe["mse_ln"][i]
                        plog[f"probe/mse_raw_stage{st}"] = probe["mse_raw"][i]
                    wandb_run.log(plog, step=global_step)

            global_step += 1
        else:
            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                path = os.path.join(args.output, f"student_ep{epoch+1}.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "student": student.state_dict(),
                    "adapters": adapters.state_dict(),
                    "ema": ema_state,
                    "args": vars(args),
                }, path)
                logger.info(f"saved {path}")
            continue
        break  # broke out of inner loop because of max-steps

    # Final probe so every sweep run ends with a comparable metric.
    if probe_batches:
        probe = run_probe(teacher, student, adapters, align,
                          probe_batches, device, args.amp)
        probe["step"] = global_step
        probe["epoch"] = args.epochs - 1
        probe_history.append(probe)
        logger.info(
            f"  FINAL PROBE step={global_step} "
            f"cos=[{','.join(f'{c:.3f}' for c in probe['cos'])}] "
            f"mse_ln=[{','.join(f'{m:.4f}' for m in probe['mse_ln'])}]"
        )
        if wandb_run is not None:
            flog = {}
            for i, st in enumerate(align):
                flog[f"probe/final_cos_stage{st}"] = probe["cos"][i]
                flog[f"probe/final_mse_ln_stage{st}"] = probe["mse_ln"][i]
            wandb_run.log(flog, step=global_step)

    final = os.path.join(args.output, "student_final.pth")
    torch.save({
        "student": student.state_dict(),
        "adapters": adapters.state_dict(),
        "ema": ema_state,
        "args": vars(args),
    }, final)
    with open(os.path.join(args.output, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if probe_history:
        with open(os.path.join(args.output, "probe.json"), "w") as f:
            json.dump(probe_history, f, indent=2)
        with open(os.path.join(args.output, "summary.json"), "w") as f:
            json.dump({
                "args": vars(args),
                "final_probe": probe_history[-1],
                "total_steps": global_step,
                "wallclock_s": time.time() - t0,
            }, f, indent=2)
    logger.info(f"done. final={final}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
