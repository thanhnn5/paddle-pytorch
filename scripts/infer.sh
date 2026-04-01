#!/bin/bash


python3 tools/infer_det.py -c configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    -o Global.pretrained_model=output/PP-OCRv5_convnext_det_unfreeze/best.pth \
    Global.infer_img="data/ocr_test_20251127/images/" \
    Global.output_dir="output/predictions" \
    PostProcess.box_thresh=0.6 \
    Global.limit_side_len=1280 \
    Global.limit_type="max"
