#!/bin/bash
# Short saliency-weighted distillation probe: 5 trials × N steps each.
# Goal: pick (alpha, beta, power) for the long run, by comparing the
# held-out probe at the end of each trial.
#
# Logs (tail-friendly while running):
#   $OUT/sweep.log                       overall progress + per-trial dividers
#   $OUT/<trial>/stdout.log              full python stdout
#   $OUT/<trial>/train.log               train.py's structured log
#   $OUT/<trial>/summary.json            single-run summary (final probe)
#
# Usage:  ./scripts/sal_probe.sh
# Env knobs (all optional):
#   IMAGES, TEACHER_CKPT, TEACHER_CFG, MAX_STEPS, BATCH_SIZE, IMG_SIZE,
#   LR, ALIGN_STAGES, COSINE_WEIGHT, OUT, SWEEP_TAG

set -euo pipefail

IMAGES=${IMAGES:-/home/jupyter/workspace/data/images}
TEACHER_CKPT=${TEACHER_CKPT:-output/PP-OCRv5_convnext_det_unfreeze/best.pth}
TEACHER_CFG=${TEACHER_CFG:-configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml}
MAX_STEPS=${MAX_STEPS:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
IMG_SIZE=${IMG_SIZE:-384}
LR=${LR:-1e-3}
ALIGN_STAGES=${ALIGN_STAGES:-2 3}
COSINE_WEIGHT=${COSINE_WEIGHT:-0.0}
SWEEP_TAG=${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-output/sal_probe/$SWEEP_TAG}

mkdir -p "$OUT"
LOG="$OUT/sweep.log"
echo "sal_probe sweep_tag=$SWEEP_TAG" | tee "$LOG"
echo "out      = $OUT"        | tee -a "$LOG"
echo "images   = $IMAGES"      | tee -a "$LOG"
echo "teacher  = $TEACHER_CKPT" | tee -a "$LOG"
echo "tcfg     = $TEACHER_CFG"  | tee -a "$LOG"
echo "steps    = $MAX_STEPS  batch=$BATCH_SIZE  img=$IMG_SIZE  lr=$LR" | tee -a "$LOG"
echo                          | tee -a "$LOG"

# Trial table: name | extra-args (empty = saliency off, baseline)
TRIALS=(
    "off               |"
    "a05_b10_p1.0      |--saliency --saliency-alpha 0.05 --saliency-beta 10 --saliency-power 1.0"
    "a05_b10_p0.5      |--saliency --saliency-alpha 0.05 --saliency-beta 10 --saliency-power 0.5"
    "a05_b20_p1.0      |--saliency --saliency-alpha 0.05 --saliency-beta 20 --saliency-power 1.0"
    "a10_b10_p1.0      |--saliency --saliency-alpha 0.10 --saliency-beta 10 --saliency-power 1.0"
)

idx=0
for row in "${TRIALS[@]}"; do
    idx=$((idx + 1))
    name=$(echo "$row" | cut -d'|' -f1 | xargs)
    extra=$(echo "$row" | cut -d'|' -f2- | sed 's/^ *//')

    trial_dir="$OUT/$name"
    mkdir -p "$trial_dir"
    stdout_log="$trial_dir/stdout.log"

    sal_tc=()
    if [ -n "$extra" ]; then
        sal_tc=(--teacher-config "$TEACHER_CFG")
    fi

    echo "==========================================" | tee -a "$LOG"
    echo "[$idx/${#TRIALS[@]}] $name  extra=[$extra]"  | tee -a "$LOG"
    echo "  out: $trial_dir"                            | tee -a "$LOG"
    echo "  log: $stdout_log"                           | tee -a "$LOG"
    echo "==========================================" | tee -a "$LOG"

    python tools/distill/train.py \
        --images "$IMAGES" \
        --teacher-ckpt "$TEACHER_CKPT" \
        "${sal_tc[@]}" \
        --output "$trial_dir" \
        --student-size femto \
        --img-size "$IMG_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --num-workers 8 \
        --epochs 999 --max-steps "$MAX_STEPS" --warmup-epochs 0 \
        --lr "$LR" \
        --align-stages $ALIGN_STAGES \
        --cosine-weight "$COSINE_WEIGHT" \
        --device auto --amp \
        --log-every 25 --probe-every 50 --probe-count 64 \
        --save-every 999 --seed 0 \
        $extra > "$stdout_log" 2>&1 || {
            echo "  [FAIL] trial $name — see $stdout_log" | tee -a "$LOG"
            continue
        }

    # Pull the key lines into the sweep log for at-a-glance check.
    echo "--- key lines from $name ---" | tee -a "$LOG"
    grep -E "saliency ON|PROBE|FINAL PROBE|sal=" "$stdout_log" | tail -10 | tee -a "$LOG"
    echo                                | tee -a "$LOG"
done

echo "==========================================" | tee -a "$LOG"
echo "SWEEP RESULTS ($SWEEP_TAG)"                  | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
python tools/distill/compare.py "$OUT/*/summary.json" --sort-by cos_last 2>&1 | tee -a "$LOG"
