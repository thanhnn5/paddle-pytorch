#!/bin/bash

python3 tools/eval.py -c configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    -o Global.pretrained_model=output/PP-OCRv5_convnext_det_unfreeze/best.pth \
    Eval.dataset.data_dir=data/ocr_test_20251127/images \
    Eval.dataset.label_file_list='[data/ocr_test_20251127/test.txt]' \
    PostProcess.box_thresh=0.6 \
    Global.limit_side_len=1280 \
    Global.limit_type="max"
