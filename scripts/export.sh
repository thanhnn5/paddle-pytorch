#!/bin/bash

# Export with fix size (this could be faster to inference)
python tools/export.py -c configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml --type onnx \
  -o Global.pretrained_model=weights/dinov3/convnext_det_unfreeze.pth \
     Export.export_dir=output/export_convnext_det \
     "Export.export_shape=[1,3,1280,704]"


# Inference with the exported model
python tools/infer/predict_det.py --det_model_dir=output/export_convnext_det \
  --image_dir=/Users/thanhnn5/Downloads/pod/pod-128.jpg \
  --output=output/predictions/exported/ \
  --det_limit_side_len=1280 \
  --use_gpu=True
