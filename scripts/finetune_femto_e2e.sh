#!/bin/bash
# End-to-end (single-phase) Stage-3 fine-tune with feature + logit distillation.
#
# Differs from scripts/finetune_femto_mobile.sh (two-phase) in two ways:
#
#   1. NO Phase-1 head warmup.  The classic warmup was needed because random
#      neck+head emit noise gradients that perturb the distilled backbone in
#      early epochs.  With logit distillation enabled (AuxDistill.logit_weight
#      > 0), the teacher's per-pixel text-probability map supervises the
#      head from epoch 1, so the head isn't producing pure noise even when
#      it starts random.  Joint training works end-to-end.
#
#   2. Logit distillation ON in addition to feature distillation.  The
#      teacher's full BaseModel is loaded so we can read its DBHead prob
#      map.  Per-pixel MSE between student and teacher prob maps pulls the
#      head toward teacher-calibrated decisions over ALL pixels (including
#      the ~85-95 % unlabeled background where detection labels are silent).
#      This is the lever for closing the student/teacher precision gap.
#
# Knobs (env vars, all optional):
#   $DISTILL_CKPT    Stage-2 distilled backbone ckpt
#                    (default: output/distill_convnext_femto_long/student_final.pth)
#   $USE_EMA         1 = extract EMA backbone (default), 0 = raw student
#   $EXTRACTED       intermediate backbone-only .pth path
#   $OUTPUT_DIR      output directory
#   $EPOCHS          total epochs   (default 75 — matches old P1+P2 sum)
#   $LR              base LR        (default 5e-4)
#   $FEAT_WEIGHT     feature-distillation weight  (default 5.0)
#   $LOGIT_WEIGHT    logit-distillation weight    (default 5.0 — start equal to feat)
#   $CONFIG          student detection config YAML
#                    (default: PP-OCRv5_convnextv2_femto_mobile_det.yml)
#
# Usage:
#   ./scripts/finetune_femto_e2e.sh
#   LOGIT_WEIGHT=10 ./scripts/finetune_femto_e2e.sh
#   CONFIG=configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_balanced_det.yml \
#     ./scripts/finetune_femto_e2e.sh

set -euo pipefail

DISTILL_CKPT=${DISTILL_CKPT:-output/distill_convnext_femto_long/student_final.pth}
USE_EMA=${USE_EMA:-1}
EXTRACTED=${EXTRACTED:-weights/distilled/femto_backbone.pth}
OUTPUT_DIR=${OUTPUT_DIR:-output/PP-OCRv5_convnextv2_femto_e2e_det}
EPOCHS=${EPOCHS:-75}
LR=${LR:-0.0005}
FEAT_WEIGHT=${FEAT_WEIGHT:-5.0}
LOGIT_WEIGHT=${LOGIT_WEIGHT:-5.0}
CONFIG=${CONFIG:-configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_mobile_det.yml}
TEACHER_CONFIG=${TEACHER_CONFIG:-configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml}
TEACHER_CKPT=${TEACHER_CKPT:-weights/dinov3/convnext_det_unfreeze.pth}

if [ ! -f "$DISTILL_CKPT" ]; then
    echo "distill ckpt not found: $DISTILL_CKPT" >&2
    exit 1
fi

# 1) Convert Stage-2 distill ckpt -> BaseModel-compatible backbone weights.
mkdir -p "$(dirname "$EXTRACTED")"
EXTRACT_ARGS=(--ckpt "$DISTILL_CKPT" --out "$EXTRACTED")
if [ "$USE_EMA" = "1" ]; then
    EXTRACT_ARGS+=(--use-ema)
fi
echo "[1/2] extract_backbone -> $EXTRACTED"
python tools/distill/extract_backbone.py "${EXTRACT_ARGS[@]}"

# 2) End-to-end Stage-3 fine-tune with feature + logit distillation.
echo "[2/2] e2e fine-tune ($EPOCHS epochs, feat=$FEAT_WEIGHT, logit=$LOGIT_WEIGHT) -> $OUTPUT_DIR"
python3 tools/train.py \
    -c "$CONFIG" \
    -o \
    Global.use_wandb=true \
    Global.wandb.project=AddressDINOv3 \
    Global.wandb.name=PPOCRv5-ConvNeXtV2-femto-e2e-stage3 \
    Global.pretrained_model="$EXTRACTED" \
    Global.output_dir="$OUTPUT_DIR" \
    Global.save_model_dir="$OUTPUT_DIR" \
    Global.epoch_num=$EPOCHS \
    Global.print_batch_step=50 \
    Global.eval_batch_step='[0,800]' \
    Optimizer.lr=$LR \
    Optimizer.freeze_backbone=false \
    Optimizer.backbone_lr_mult=0.1 \
    Optimizer.neck_lr_mult=1.0 \
    Optimizer.head_lr_mult=1.0 \
    AuxDistill.enabled=true \
    AuxDistill.teacher_ckpt="$TEACHER_CKPT" \
    AuxDistill.teacher_config="$TEACHER_CONFIG" \
    AuxDistill.feat_weight=$FEAT_WEIGHT \
    AuxDistill.logit_weight=$LOGIT_WEIGHT \
    Train.loader.batch_size_per_card=32 \
    Train.dataset.data_dir=data/det_v4/ \
    Train.dataset.label_file_list='[./data/det_v4/train.txt]' \
    Eval.dataset.data_dir=data/det_v4/ \
    Eval.dataset.label_file_list='[./data/det_v4/val.txt]' \
    Eval.dataset.transforms[2].DetResizeForTest.limit_side_len=1280 \
    Eval.dataset.transforms[2].DetResizeForTest.limit_type=max \
    Eval.dataset.transforms[2].DetResizeForTest.keep_ratio=true

echo "done. final weights: $OUTPUT_DIR/best.pth"
