#!/bin/bash


python3 tools/infer/pytorch/infer_det.py -c configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml \
    -o Global.pretrained_model=weights/dinov3/convnext_det_unfreeze.pth \
    Global.infer_img="/Users/thanhnn5/Downloads/pod/pod-128.jpg" \
    Global.output_dir="output/predictions" \
    PostProcess.box_thresh=0.6 \
    Global.limit_side_len=1280 \
    Global.limit_type="max"


# Inference with the converted MNN model (Metal backend, low precision = fp16)
python tools/infer/mnn/predict_det.py \
    --det_model_dir=output/export_convnext_det \
    --image_dir=/Users/thanhnn5/Downloads/pod/pod-128.jpg \
    --output=output/predictions/mnn_metal/ \
    --det_limit_side_len=1280 \
    --det_limit_type=max \
    --mnn_backend=metal \
    --mnn_precision=low \
    --warmup
