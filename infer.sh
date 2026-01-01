#!/bin/bash


python3 tools/infer_det.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
    -o Global.pretrained_model=weights/mob_det_2.pth \
    Global.infer_img="/Users/thanhnn5/Workspace/Jitsu/POD/PytorchOCR/images" \
    PostProcess.box_thresh=0.7 \
    Global.limit_side_len=1280 \
    Global.limit_type="max"
