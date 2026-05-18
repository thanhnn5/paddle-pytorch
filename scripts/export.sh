#!/bin/bash

# Export with fix size (this could be faster to inference)
python tools/deployment/export_onnx.py -c configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml --type onnx \
  -o Global.pretrained_model=weights/dinov3/convnext_det_unfreeze.pth \
     Export.export_dir=output/export_convnext_det \
     "Export.export_shape=[1,3,1280,704]"


# Inference with the exported model
python tools/infer/onnx/predict_det.py \
  --det_model_dir=output/export_convnext_det \
  --image_dir=/Users/thanhnn5/Downloads/pod/pod-128.jpg \
  --output=output/predictions/exported/ \
  --det_limit_side_len=1280 \
  --use_gpu=True


# Convert exported ONNX model to MNN
mnnconvert -f ONNX \
  --modelFile output/export_convnext_det/model.onnx \
  --MNNModel output/export_convnext_det/model.mnn \
  --fp16 --transformerFuse=0


# Inference with the converted MNN model
python tools/infer/mnn/predict_det.py \
  --det_model_dir=output/export_convnext_det \
  --image_dir=/Users/thanhnn5/Downloads/pod/pod-138.jpg \
  --output=output/predictions/exported/ \
  --mnn_backend=metal --mnn_precision=low
