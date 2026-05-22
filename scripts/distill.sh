#!/bin/bash
# Stage-2 feature distillation: ConvNeXt-V2-pico student <- DINOv3-ConvNeXt-tiny teacher.
# See docs/convnext_distillation.md (design) and
# docs/distill_saliency_weighted_loss.md (saliency weighting + sweep results).
#
# Defaults reflect the validated sweep results:
#   batch=256, lr=1e-3, cosine-weight=0      (bigbatch_20260518 sweep)
#   saliency on, α=0.05, β=10, power=0.3     (refine_20260519 sweep)

set -euo pipefail

python3 tools/distill/train.py \
    --images /home/jupyter/workspace/data/images \
    --teacher-ckpt output/PP-OCRv5_convnext_det_unfreeze/best.pth \
    --teacher-config configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    --output output/distill_convnext_pico_sal_long \
    --student-size pico \
    --img-size 384 \
    --batch-size 128 \
    --num-workers 8 \
    --epochs 100 \
    --warmup-epochs 5 \
    --lr 1e-3 \
    --weight-decay 0.05 \
    --grad-clip 1.0 \
    --align-stages 2 3 \
    --cosine-weight 0.0 \
    --saliency \
    --saliency-alpha 0.05 \
    --saliency-beta 10 \
    --saliency-power 0.3 \
    --ema-decay 0.9995 \
    --device auto \
    --amp \
    --log-every 50 \
    --probe-every 500 \
    --probe-count 128 \
    --save-every 5 \
    --seed 0
