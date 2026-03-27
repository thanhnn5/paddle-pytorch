#!/bin/bash

python3 tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_dinov3_det.yml \
    -o Train.dataset.data_dir=./ocr_det_dataset_examples \
    Train.dataset.label_file_list='[./ocr_det_dataset_examples/train.txt]' \
    Eval.dataset.data_dir=./ocr_det_dataset_examples \
    Eval.dataset.label_file_list='[./ocr_det_dataset_examples/val.txt]'
