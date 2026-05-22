#!/bin/bash
# Stage-2 saliency-weighted distillation on a mixed image pool:
#   POD + COCO-Text + ICDAR2015
#
# The HF datasets are downloaded + cached to ~/.cache/huggingface on first run
# (~16k + 1.5k = ~17.5k extra scene-text images, a few hundred MB total).
# Subsequent runs reuse the cache. Set HF_DATASETS_CACHE / --hf-cache-dir to
# relocate.
#
# Mixing weights are sampling probabilities, not epoch proportions. Defaults:
#   50% POD          (preserve target distribution)
#   30% COCO-Text    (43k labeled scene-text images, English)
#   20% ICDAR2015    (1.5k high-quality incidental scene text)
#
# Override via env vars:
#   POD_W=0.4 COCO_W=0.4 ICDAR_W=0.2 ./scripts/distill_saliency_mix.sh

set -euo pipefail

POD_W=${POD_W:-0.5}
COCO_W=${COCO_W:-0.3}
ICDAR_W=${ICDAR_W:-0.2}

python3 tools/distill/train.py \
    --images data/det_v4/images \
    --teacher-ckpt weights/dinov3/convnext_det_unfreeze.pth \
    --teacher-config configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    --saliency \
    --saliency-alpha 0.05 \
    --saliency-beta 10.0 \
    --saliency-power 0.3 \
    --hf-dataset howard-hou/COCO-Text \
    --hf-dataset-splits train,validation \
    --hf-dataset dlxjj/ICDAR2015 \
    --hf-dataset-splits train,test \
    --pod-weight "$POD_W" \
    --hf-weight "$COCO_W" \
    --hf-weight "$ICDAR_W" \
    --output output/distill_convnext_pico_saliency_mix \
    --student-size pico \
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
