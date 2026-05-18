#!/bin/bash
# Quick hyperparameter sweep for Stage-2 distillation. Each trial runs for
# MAX_STEPS short steps so the whole grid finishes in <1 hour on a single GPU.
# After all trials, prints a sorted comparison table by held-out probe cosine.
#
# Edit the grids below, then run: ./scripts/distill_sweep.sh
# Aggregate results later with:    python tools/distill/compare.py output/sweep_*/summary.json

set -euo pipefail

IMAGES=${IMAGES:-data/det_v4/images}
TEACHER=${TEACHER:-weights/dinov3/convnext_det_unfreeze.pth}
SWEEP_TAG=${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}
MAX_STEPS=${MAX_STEPS:-1500}
DEVICE=${DEVICE:-auto}
COMMON_ARGS=(
    --images "$IMAGES"
    --teacher-ckpt "$TEACHER"
    --max-steps "$MAX_STEPS"
    --epochs 999          # high cap, we stop on max-steps
    --warmup-epochs 0     # short trials, skip warmup
    --save-every 999      # no intermediate ckpts during sweep
    --log-every 100
    --probe-every 250
    --probe-count 64
    --num-workers 4
    --device "$DEVICE"
    --amp
)

# Grids — override via env vars as space-separated strings, e.g.
#   SEEDS="0 1 2" LRS="1e-3 2e-3" ./scripts/distill_sweep.sh
# Multi-seed runs (SEEDS="0 1 2") give 3x cost but verify that a config's win
# isn't a lucky-seed artifact. Use single seed for first-pass screening,
# multi-seed for final 2-3 candidates.
read -r -a SEEDS          <<< "${SEEDS:-0}"
read -r -a LRS            <<< "${LRS:-5e-4 1e-3 2e-3}"
read -r -a IMG_SIZES      <<< "${IMG_SIZES:-256 384}"
read -r -a COSINE_WEIGHTS <<< "${COSINE_WEIGHTS:-0.0 0.5}"
# ALIGN_STAGES uses '|' as separator since each entry is space-containing.
IFS='|' read -r -a ALIGN_STAGES <<< "${ALIGN_STAGES:-2 3|3|1 2 3}"

mkdir -p "output/sweeps/$SWEEP_TAG"
SUMMARY_GLOB="output/sweeps/$SWEEP_TAG/*/summary.json"

trial_idx=0
for seed in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
        for img in "${IMG_SIZES[@]}"; do
            for cw in "${COSINE_WEIGHTS[@]}"; do
                for stages in "${ALIGN_STAGES[@]}"; do
                    trial_idx=$((trial_idx + 1))
                    name="t${trial_idx}_lr${lr}_img${img}_cw${cw}_st${stages// /-}_s${seed}"
                    out="output/sweeps/$SWEEP_TAG/$name"
                    echo "=========================================="
                    echo "[$trial_idx] $name"
                    echo "=========================================="
                    python tools/distill/train.py \
                        "${COMMON_ARGS[@]}" \
                        --output "$out" \
                        --lr "$lr" \
                        --img-size "$img" \
                        --cosine-weight "$cw" \
                        --align-stages $stages \
                        --seed "$seed" \
                        --batch-size 32      # smaller for fast trials
                done
            done
        done
    done
done

echo
echo "=========================================="
echo "SWEEP RESULTS ($SWEEP_TAG)"
echo "=========================================="
python tools/distill/compare.py "$SUMMARY_GLOB" --sort-by cos_last
