#!/bin/bash
# Stage-2 saliency-weighted distillation on a mixed image pool:
#   POD + ICDAR2015 + COCO (subset)
#
# COCO-Text was dropped: its image set is a subset of COCO 2017 (just with
# text annotations), so it overlapped redundantly with detection-datasets/coco.
# We use a 40k subset of COCO instead — comparable diversity to the original
# COCO-Text contribution but in the broader natural-scene distribution.
#
# HF datasets are downloaded + cached to ~/.cache/huggingface on first run.
# Set HF_DATASETS_CACHE / --hf-cache-dir to relocate. Cache sizes:
#   ICDAR2015                ~140 MB    (1.5k incidental scene text)
#   detection-datasets/coco  ~20 GB     (downloads full 117k; we --select 40k)
#
# Mixing weights are sampling probabilities, not epoch proportions. Small
# datasets get revisited more often per training step. Defaults:
#   70% POD          (preserve target distribution)
#   25% COCO         (40k natural-scene images; regularizes the backbone away
#                     from POD-only feature collapse. Teacher DBHead is ~0
#                     on most of these, so saliency-weighted loss reduces to
#                     uniform low-weight MSE — the goal is feature-distribution
#                     coverage, not text supervision)
#    5% ICDAR2015    (1.5k high-quality scene text; small but high-value)
#
# Override via env vars:
#   POD_W=0.6 COCO_W=0.3 ICDAR_W=0.1 ./scripts/distill_saliency_mix.sh

set -euo pipefail

POD_W=${POD_W:-0.70}
COCO_W=${COCO_W:-0.25}
ICDAR_W=${ICDAR_W:-0.05}
COCO_MAX_SAMPLES=${COCO_MAX_SAMPLES:-40000}

python3 tools/distill/train.py \
    --images data/det_v4/images \
    --teacher-ckpt weights/dinov3/convnext_det_unfreeze.pth \
    --teacher-config configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    --saliency \
    --saliency-alpha 0.05 \
    --saliency-beta 10.0 \
    --saliency-power 0.3 \
    --hf-dataset dlxjj/ICDAR2015 \
    --hf-dataset-splits train,test \
    --hf-dataset detection-datasets/coco \
    --hf-dataset-splits train \
    --hf-max-samples "$COCO_MAX_SAMPLES" \
    --pod-weight "$POD_W" \
    --hf-weight "$ICDAR_W" \
    --hf-weight "$COCO_W" \
    --output output/distill_convnext_femto_saliency_mix \
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
