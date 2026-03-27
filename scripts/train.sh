#!/bin/bash

python3 tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_dinov3_det.yml \
    -o Global.epoch_num=50 \
    Global.print_batch_step=50 \
    Train.dataset.loader.batch_size_per_card=8 \
    Train.dataset.data_dir=data/ocr_det_mask_3-1/ \
    Train.dataset.label_file_list='[./data/ocr_det_mask_3-1/train.txt]' \
    Eval.dataset.data_dir=data/ocr_det_mask_3-1/ \
    Eval.dataset.label_file_list='[./data/ocr_det_mask_3-1/val.txt]'
