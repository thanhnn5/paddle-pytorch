# PytorchOCR

- Repo contains code for training/eval/infer PaddleOCR models with PyTorch backend.

### train
```sh
CUDA_VISIBLE_DEVICES=0 \
  python tools/train.py \
  -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nnodes=1 --nproc_per_node=4 \
  tools/train.py \
  -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml
```


### eval
```sh
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.checkpoints=xxx.pth
```


### infer
```sh
python tools/infer_rec.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.pretrained_model=xxx.pth
```

### export
```sh
# Export to pytorch
# Ensure the PaddleOCR model is exported with the old format
python tools/export.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml -o Global.pretrained_model=xxx.pth


# Export to torch executorch, check convert_executorch.py for more details
# Static shape is more stable and faster
python convert_executorch.py
```


### predict
```sh
# det + cls + rec
python tools/infer/predict_system.py \
  --det_model_dir=path/to/det/export_dir  \
  --cls_model_dir=path/to/cls/export_dir  \
  --rec_model_dir=path/to/rec/export_dir  \
  --image_dir=doc/imgs/1.jpg \
  --use_angle_cls=true

# det
python tools/infer/predict_det.py \
  --det_model_dir=path/to/det/export_dir \
  --image_dir=doc/imgs/1.jpg

# cls
python tools/infer/predict_cls.py \
  --cls_model_dir=path/to/cls/export_dir \
  --image_dir=doc/imgs/1.jpg

# rec
python tools/infer/predict_rec.py \
  --rec_model_dir=path/to/rec/export_dir \
  --image_dir=doc/imgs_words/en/word_1.png
```

ref:

1. https://github.com/PaddlePaddle/PaddleOCR
2. https://github.com/frotms/PaddleOCR2Pytorch
3. https://github.com:WenmuZhou/PytorchOCR