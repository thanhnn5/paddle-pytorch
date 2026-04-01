#!/bin/bash

python3 tools/train.py \
    -c configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    -o \
    Global.use_wandb=true \
    Global.wandb.project=AddressDINOv3 \
    Global.wandb.name=PPOCRv5-DINOv3-ConvNext_backbone-lr_no-mask_unfreeze-bb \
    Global.pretrained_model=output/PP-OCRv5_convnext_det/best.pth \
    Global.output_dir=output/PP-OCRv5_convnext_det_unfreeze \
    Global.save_model_dir=output/PP-OCRv5_convnext_det_unfreeze \
    Global.epoch_num=50 \
    Global.print_batch_step=50 \
    Global.eval_batch_step='[0,800]' \
    Optimizer.lr=0.0001 \
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


    # Optimizer.freeze_backbone=true \
