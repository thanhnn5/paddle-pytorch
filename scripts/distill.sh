#!/bin/bash
# Stage-2 feature distillation: ConvNeXt-V2-femto student <- DINOv3-ConvNeXt-tiny teacher.
# See docs/convnext_distillation.md for the design.

set -euo pipefail

python3 tools/distill/train.py \
    --images data/det_v4/images \
    --teacher-ckpt weights/dinov3/convnext_det_unfreeze.pth \
    --output output/distill_convnext_femto \
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
