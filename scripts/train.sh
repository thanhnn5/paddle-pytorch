#!/bin/bash

python3 tools/train.py \
    -c configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    -o \
    Global.use_wandb=true \
    Global.wandb.project=AddressDINOv3 \
    Global.wandb.name=PPOCRv5-DINOv3_ConvNext \
    Global.epoch_num=50 \
    Global.print_batch_step=50 \
    Optimizer.lr=0.0001 \
    Train.dataset.loader.batch_size_per_card=8 \
    Train.dataset.data_dir=data/ocr_det_mask_3-1/ \
    Train.dataset.label_file_list='[./data/ocr_det_mask_3-1/train.txt]' \
    Eval.dataset.data_dir=data/ocr_det_mask_3-1/ \
    Eval.dataset.label_file_list='[./data/ocr_det_mask_3-1/val.txt]' \
    Eval.dataset.transforms[2].DetResizeForTest.limit_side_len=1280 \
    Eval.dataset.transforms[2].DetResizeForTest.limit_type=max \
    Eval.dataset.transforms[2].DetResizeForTest.keep_ratio=true
