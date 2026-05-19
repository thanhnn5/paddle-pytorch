#!/bin/bash
# Stage-2 feature distillation with teacher-saliency weighting (Option B).
# See docs/distill_saliency_weighted_loss.md.
#
# Differs from scripts/distill.sh in three things:
#   - Loads the FULL BaseModel teacher (backbone + LKPAN + DBHead) via
#     --teacher-config, so the DBHead sigmoid output is available as a
#     per-pixel text-probability map.
#   - --saliency enables per-pixel weighting of the per-stage MSE loss.
#   - Defaults: alpha=0.05, beta=10 — calibrated for POD's sparse prob maps
#     (mean ~0.03). Yields ~38% gradient share on text regions vs ~0.3% for
#     uniform. See the doc for the math.

set -euo pipefail

python3 tools/distill/train.py \
    --images data/det_v4/images \
    --teacher-ckpt weights/dinov3/convnext_det_unfreeze.pth \
    --teacher-config configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    --saliency \
    --saliency-alpha 0.05 \
    --saliency-beta 10.0 \
    --saliency-power 1.0 \
    --output output/distill_convnext_femto_saliency \
    --student-size femto \
    --img-size 384 \
    --batch-size 64 \
    --num-workers 8 \
    --epochs 100 \
    --warmup-epochs 5 \
    --lr 1e-3 \
    --weight-decay 0.05 \
    --grad-clip 1.0 \
    --align-stages 2 3 \
    --cosine-weight 0.5 \
    --ema-decay 0.9995 \
    --device auto \
    --amp \
    --log-every 50 \
    --save-every 5 \
    --seed 0
