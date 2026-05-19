#!/bin/bash
# Two-phase Stage-3 detection fine-tune for the distilled ConvNeXt-V2-femto
# backbone paired with RSEFPN+DBHead (PP-OCRv5-mobile style).
#
#   Phase 1 (25 ep): backbone FROZEN. Only RSEFPN+DBHead train, letting the
#                    randomly-initialized neck/head stabilize against the
#                    fixed distilled features without dragging them around.
#   Phase 2 (50 ep): backbone UNFROZEN with backbone_lr_mult=0.1. Joint
#                    fine-tune; loads phase-1 weights (backbone+neck+head)
#                    via pretrained_model so neck/head keep their warm state.
#
# Inputs (env vars, all optional):
#   $DISTILL_CKPT  Stage-2 final ckpt (default
#                  output/distill_convnext_femto_long/student_final.pth)
#   $USE_EMA       1 = extract EMA backbone (default), 0 = raw student
#   $EXTRACTED     intermediate backbone-only .pth path
#   $OUTPUT_DIR    parent dir; phase outputs go in $OUTPUT_DIR/phase{1,2}
#   $P1_EPOCHS     phase-1 epoch count (default 25)
#   $P2_EPOCHS     phase-2 epoch count (default 50)
#   $P1_LR         phase-1 lr  (default 1e-3 — head is fully random, can take more)
#   $P2_LR         phase-2 lr  (default 5e-4 — joint fine-tune)
#
# Usage:
#   ./scripts/finetune_femto_mobile.sh
#   DISTILL_CKPT=output/distill_other/student_ep80.pth ./scripts/finetune_femto_mobile.sh

set -euo pipefail

DISTILL_CKPT=${DISTILL_CKPT:-output/distill_convnext_femto_long/student_final.pth}
USE_EMA=${USE_EMA:-1}
EXTRACTED=${EXTRACTED:-weights/distilled/femto_backbone.pth}
OUTPUT_DIR=${OUTPUT_DIR:-output/PP-OCRv5_convnextv2_femto_mobile_det}
P1_EPOCHS=${P1_EPOCHS:-25}
P2_EPOCHS=${P2_EPOCHS:-50}
P1_LR=${P1_LR:-0.001}
P2_LR=${P2_LR:-0.0005}

P1_DIR="$OUTPUT_DIR/phase1_frozen"
P2_DIR="$OUTPUT_DIR/phase2_joint"
CONFIG=configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_mobile_det.yml

if [ ! -f "$DISTILL_CKPT" ]; then
    echo "distill ckpt not found: $DISTILL_CKPT" >&2
    exit 1
fi

# 0) Convert distill ckpt -> BaseModel-compatible flat state_dict (backbone.*)
mkdir -p "$(dirname "$EXTRACTED")"
EXTRACT_ARGS=(--ckpt "$DISTILL_CKPT" --out "$EXTRACTED")
if [ "$USE_EMA" = "1" ]; then
    EXTRACT_ARGS+=(--use-ema)
fi
echo "[0/2] extract_backbone -> $EXTRACTED"
python tools/distill/extract_backbone.py "${EXTRACT_ARGS[@]}"

# 1) Phase 1: frozen backbone, train neck+head only.
#    Higher LR (no backbone drift to worry about), shorter schedule.
echo "[1/2] phase 1 (frozen backbone, $P1_EPOCHS epochs) -> $P1_DIR"
python3 tools/train.py \
    -c "$CONFIG" \
    -o \
    Global.use_wandb=true \
    Global.wandb.project=AddressDINOv3 \
    Global.wandb.name=PPOCRv5-ConvNeXtV2-femto-mobile-stage3-p1-frozen \
    Global.pretrained_model="$EXTRACTED" \
    Global.output_dir="$P1_DIR" \
    Global.save_model_dir="$P1_DIR" \
    Global.epoch_num=$P1_EPOCHS \
    Global.print_batch_step=50 \
    Global.eval_batch_step='[0,800]' \
    Optimizer.lr=$P1_LR \
    Optimizer.freeze_backbone=true \
    Optimizer.neck_lr_mult=1.0 \
    Optimizer.head_lr_mult=1.0 \
    Train.loader.batch_size_per_card=32 \
    Train.dataset.data_dir=data/det_v4/ \
    Train.dataset.label_file_list='[./data/det_v4/train.txt]' \
    Eval.dataset.data_dir=data/det_v4/ \
    Eval.dataset.label_file_list='[./data/det_v4/val.txt]' \
    Eval.dataset.transforms[2].DetResizeForTest.limit_side_len=1280 \
    Eval.dataset.transforms[2].DetResizeForTest.limit_type=max \
    Eval.dataset.transforms[2].DetResizeForTest.keep_ratio=true

P1_BEST="$P1_DIR/lastest.pth"
if [ ! -f "$P1_BEST" ]; then
    # Fallback: some configs save best as `best_accuracy.pth` or similar.
    P1_BEST=$(ls -t "$P1_DIR"/*.pth 2>/dev/null | head -1)
    echo "[note] phase-1 best.pth not found; using $P1_BEST" >&2
fi
if [ ! -f "$P1_BEST" ]; then
    echo "no phase-1 checkpoint produced in $P1_DIR" >&2
    exit 1
fi

# 2) Phase 2: unfreeze backbone, joint fine-tune from phase-1 weights.
#    backbone_lr_mult=0.1 protects the distilled features.
echo "[2/2] phase 2 (unfrozen joint, $P2_EPOCHS epochs) -> $P2_DIR"
echo "[2/2] resuming from $P1_BEST"
python3 tools/train.py \
    -c "$CONFIG" \
    -o \
    Global.use_wandb=true \
    Global.wandb.project=AddressDINOv3 \
    Global.wandb.name=PPOCRv5-ConvNeXtV2-femto-mobile-stage3-p2-joint \
    Global.pretrained_model="$P1_BEST" \
    Global.output_dir="$P2_DIR" \
    Global.save_model_dir="$P2_DIR" \
    Global.epoch_num=$P2_EPOCHS \
    Global.print_batch_step=50 \
    Global.eval_batch_step='[0,800]' \
    Optimizer.lr=$P2_LR \
    Optimizer.freeze_backbone=false \
    Optimizer.backbone_lr_mult=0.1 \
    Optimizer.neck_lr_mult=1.0 \
    Optimizer.head_lr_mult=1.0 \
    Train.loader.batch_size_per_card=32 \
    Train.dataset.data_dir=data/det_v4/ \
    Train.dataset.label_file_list='[./data/det_v4/train.txt]' \
    Eval.dataset.data_dir=data/det_v4/ \
    Eval.dataset.label_file_list='[./data/det_v4/val.txt]' \
    Eval.dataset.transforms[2].DetResizeForTest.limit_side_len=1280 \
    Eval.dataset.transforms[2].DetResizeForTest.limit_type=max \
    Eval.dataset.transforms[2].DetResizeForTest.keep_ratio=true

echo "done. final weights: $P2_DIR/best.pth"
