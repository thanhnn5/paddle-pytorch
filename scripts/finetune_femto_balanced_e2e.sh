#!/bin/bash
# Single-phase end-to-end Stage-3 fine-tune of the distilled ConvNeXt-V2-femto
# backbone with the LKPAN-96 + PFHL-small head (balanced config), plus:
#
#   - AuxDistill (feature-only, weight 10.0) — anchors backbone to teacher
#     features throughout training.
#   - Backbone LR multiplier 0.25 (vs the usual 0.1) — gives the backbone
#     more room to refine itself for POD, since the AuxDistill anchor
#     prevents excessive drift.
#   - FlatCosineLR schedule — long peak-LR plateau (~half of training) before
#     cosine decay. Pairs well with the AuxDistill regularizer: backbone has
#     time at peak LR to adapt while the anchor restrains it.
#
# Defaults (override via env vars at invocation):
#   $DISTILL_CKPT      Stage-2 distilled student ckpt
#   $USE_EMA           1 = extract EMA weights, 0 = raw student (default 1)
#   $EXTRACTED         intermediate backbone-only .pth path
#   $OUTPUT_DIR        output directory
#   $EPOCHS            total epochs (default 100)
#   $LR                base learning rate (default 5e-4)
#   $BACKBONE_LR_MULT  backbone LR multiplier (default 0.25 = 1/4)
#   $AUX_WEIGHT        AuxDistill feature loss weight (default 10.0)
#   $FLAT_EPOCH        epoch at which cosine decay starts (default epochs/2)
#   $LR_GAMMA          minimum LR as fraction of peak (default 0.05)
#
# Usage:
#   ./scripts/finetune_femto_balanced_e2e.sh
#   AUX_WEIGHT=15 BACKBONE_LR_MULT=0.3 ./scripts/finetune_femto_balanced_e2e.sh

set -euo pipefail

DISTILL_CKPT=${DISTILL_CKPT:-output/distill_convnext_femto_long/student_final.pth}
USE_EMA=${USE_EMA:-1}
EXTRACTED=${EXTRACTED:-weights/distilled/femto_backbone.pth}
OUTPUT_DIR=${OUTPUT_DIR:-output/PP-OCRv5_convnextv2_femto_balanced_e2e}
EPOCHS=${EPOCHS:-100}
LR=${LR:-0.0005}
BACKBONE_LR_MULT=${BACKBONE_LR_MULT:-0.25}
AUX_WEIGHT=${AUX_WEIGHT:-10.0}
FLAT_EPOCH=${FLAT_EPOCH:-$((EPOCHS / 2))}
LR_GAMMA=${LR_GAMMA:-0.05}

CONFIG=configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_balanced_det.yml
TEACHER_CKPT=${TEACHER_CKPT:-weights/dinov3/convnext_det_unfreeze.pth}

if [ ! -f "$DISTILL_CKPT" ]; then
    echo "distill ckpt not found: $DISTILL_CKPT" >&2
    exit 1
fi

# 1) Convert Stage-2 distill ckpt -> BaseModel-compatible backbone weights
mkdir -p "$(dirname "$EXTRACTED")"
EXTRACT_ARGS=(--ckpt "$DISTILL_CKPT" --out "$EXTRACTED")
if [ "$USE_EMA" = "1" ]; then
    EXTRACT_ARGS+=(--use-ema)
fi
echo "[1/2] extract_backbone -> $EXTRACTED"
python tools/distill/extract_backbone.py "${EXTRACT_ARGS[@]}"

# 2) E2E Stage-3 with AuxDistill + FlatCosineLR
echo "[2/2] e2e fine-tune ($EPOCHS ep, aux_w=$AUX_WEIGHT, bb_lr_mult=$BACKBONE_LR_MULT, flat_epoch=$FLAT_EPOCH)"
python3 tools/train.py \
    -c "$CONFIG" \
    -o \
    Global.use_wandb=true \
    Global.wandb.project=AddressDINOv3 \
    Global.wandb.name=PPOCRv5-ConvNeXtV2-femto-balanced-e2e-aux10-bblr0.25-flatcosine \
    Global.pretrained_model="$EXTRACTED" \
    Global.output_dir="$OUTPUT_DIR" \
    Global.save_model_dir="$OUTPUT_DIR" \
    Global.epoch_num=$EPOCHS \
    Global.print_batch_step=50 \
    Global.eval_batch_step='[0,800]' \
    Optimizer.lr=$LR \
    Optimizer.freeze_backbone=false \
    Optimizer.backbone_lr_mult=$BACKBONE_LR_MULT \
    Optimizer.neck_lr_mult=1.0 \
    Optimizer.head_lr_mult=1.0 \
    LRScheduler.name=FlatCosineLR \
    LRScheduler.warmup_epoch=2 \
    LRScheduler.flat_epoch=$FLAT_EPOCH \
    LRScheduler.cooldown_epoch=0 \
    LRScheduler.lr_gamma=$LR_GAMMA \
    AuxDistill.enabled=true \
    AuxDistill.teacher_ckpt="$TEACHER_CKPT" \
    AuxDistill.align_stages='[2,3]' \
    AuxDistill.weight=$AUX_WEIGHT \
    Train.loader.batch_size_per_card=32 \
    Train.dataset.data_dir=data/det_v4/ \
    Train.dataset.label_file_list='[./data/det_v4/train.txt]' \
    Eval.dataset.data_dir=data/det_v4/ \
    Eval.dataset.label_file_list='[./data/det_v4/val.txt]' \
    Eval.dataset.transforms[2].DetResizeForTest.limit_side_len=1280 \
    Eval.dataset.transforms[2].DetResizeForTest.limit_type=max \
    Eval.dataset.transforms[2].DetResizeForTest.keep_ratio=true

echo "done. final weights: $OUTPUT_DIR/best.pth"
