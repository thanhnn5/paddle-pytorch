#!/bin/bash
# Stage-3 detection fine-tune of the distilled ConvNeXt-V2-femto backbone.
#
# Inputs:
#   $DISTILL_CKPT  Stage-2 final ckpt (default uses
#                  output/distill_convnext_femto_long/student_final.pth)
#   $USE_EMA       1 to extract EMA weights, 0 to use raw student (default 1)
#
# Usage:
#   ./scripts/finetune_femto.sh
#   DISTILL_CKPT=output/distill_other/student_ep80.pth ./scripts/finetune_femto.sh

set -euo pipefail

DISTILL_CKPT=${DISTILL_CKPT:-output/distill_convnext_femto_long/student_final.pth}
USE_EMA=${USE_EMA:-1}
EXTRACTED=${EXTRACTED:-weights/distilled/femto_backbone.pth}
OUTPUT_DIR=${OUTPUT_DIR:-output/PP-OCRv5_convnextv2_femto_det}

if [ ! -f "$DISTILL_CKPT" ]; then
    echo "distill ckpt not found: $DISTILL_CKPT" >&2
    exit 1
fi

# 1) Convert distill ckpt -> BaseModel-compatible flat state_dict (backbone.*)
mkdir -p "$(dirname "$EXTRACTED")"
EXTRACT_ARGS=(--ckpt "$DISTILL_CKPT" --out "$EXTRACTED")
if [ "$USE_EMA" = "1" ]; then
    EXTRACT_ARGS+=(--use-ema)
fi
echo "[1/2] extract_backbone -> $EXTRACTED"
python tools/distill/extract_backbone.py "${EXTRACT_ARGS[@]}"

# 2) Stage-3 detection fine-tune. Mirrors scripts/train.sh's override pattern
#    for the POD dataset; swap data_dir / label_file_list to your POD paths.
echo "[2/2] launching tools/train.py"
python3 tools/train.py \
    -c configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_det.yml \
    -o \
    Global.use_wandb=true \
    Global.wandb.project=AddressDINOv3 \
    Global.wandb.name=PPOCRv5-ConvNeXtV2-femto-distilled-stage3 \
    Global.pretrained_model="$EXTRACTED" \
    Global.output_dir="$OUTPUT_DIR" \
    Global.save_model_dir="$OUTPUT_DIR" \
    Global.epoch_num=100 \
    Global.print_batch_step=50 \
    Global.eval_batch_step='[0,800]' \
    Optimizer.lr=0.0005 \
    Optimizer.backbone_lr_mult=0.1 \
    Optimizer.neck_lr_mult=1.0 \
    Optimizer.head_lr_mult=1.0 \
    Train.loader.batch_size_per_card=16 \
    Train.dataset.data_dir=data/det_v4/ \
    Train.dataset.label_file_list='[./data/det_v4/train.txt]' \
    Eval.dataset.data_dir=data/det_v4/ \
    Eval.dataset.label_file_list='[./data/det_v4/val.txt]' \
    Eval.dataset.transforms[2].DetResizeForTest.limit_side_len=1280 \
    Eval.dataset.transforms[2].DetResizeForTest.limit_type=max \
    Eval.dataset.transforms[2].DetResizeForTest.keep_ratio=true
