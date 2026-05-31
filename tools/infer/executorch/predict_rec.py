"""Recognition inference with an ExecuTorch .pte (coreml / xnnpack).

Mirrors tools/infer/mnn/predict_rec.py for the static [1, 3, 48, 320] export of
PP-OCRv5 mobile rec. Same two ExecuTorch caveats as predict_det.py apply:

  1. Input must be C-contiguous to match the export dim-order (we force it with
     np.ascontiguousarray). The rec preprocessing already builds a C-contiguous
     buffer, but we normalize defensively.
  2. The rec head (MultiHead) returns {'res': ctc_logits} at eval — a single
     output flattened to index 0 in the .pte; CTCLabelDecode consumes it.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

import cv2
import argparse
import math
import numpy as np
import time

import torch

from torchocr import Config
from torchocr.postprocess import build_post_process
from torchocr.utils.logging import get_logger
from torchocr.utils.utility import get_image_file_list, check_and_read

logger = get_logger()


class ExecuTorchEngine:
    def __init__(self, pte_path):
        if not os.path.exists(pte_path):
            raise Exception(f'{pte_path} is not exists')
        from executorch.runtime import Runtime
        self.method = Runtime.get().load_program(pte_path).load_method('forward')

    def run(self, image_numpy):
        x = torch.from_numpy(np.ascontiguousarray(image_numpy.astype(np.float32)))
        out = self.method.execute([x])
        outs = out if isinstance(out, (list, tuple)) else [out]
        return [t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)
                for t in outs]


class TextRecognizerET:
    """Static-shape ExecuTorch rec runner — one forward per crop, letterboxed
    into 48x320 (no max_wh_ratio bucketing, like the MNN path)."""

    def __init__(self, args):
        if args.rec_model_dir is None or not os.path.exists(args.rec_model_dir):
            raise Exception(f'args.rec_model_dir is set to {args.rec_model_dir}, but it is not exists')

        pte_path = os.path.join(args.rec_model_dir, f'model_{args.et_backend}.pte')
        config_path = os.path.join(args.rec_model_dir, 'config.yaml')

        self.engine = ExecuTorchEngine(pte_path)
        self.out_index = args.et_out_index
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(',')]
        cfg = Config(config_path).cfg
        self.postprocess_op = build_post_process(cfg['PostProcess'])

        self.img_mode = 'BGR'
        for op in cfg.get('Transforms', []):
            if 'DecodeImage' in op and op['DecodeImage']:
                self.img_mode = op['DecodeImage'].get('img_mode', 'BGR')
                break

    def resize_norm_img(self, img):
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgC, imgH, imgW = self.rec_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = min(imgW, int(math.ceil(imgH * ratio)))
        resized = cv2.resize(img, (resized_w, imgH)).astype('float32')
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized -= 0.5
        resized /= 0.5
        padded = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        return padded

    def __call__(self, img_list):
        rec_res = [['', 0.0]] * len(img_list)
        st = time.time()
        for i, img in enumerate(img_list):
            norm = self.resize_norm_img(img)[np.newaxis, :]
            preds = self.engine.run(norm)[self.out_index]
            out = self.postprocess_op({'res': preds})
            rec_res[i] = out[0]
        return rec_res, time.time() - st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rec_model_dir', type=str, required=True)
    p.add_argument('--image_dir', type=str, required=True)
    p.add_argument('--rec_image_shape', type=str, default='3,48,320')
    p.add_argument('--et_backend', type=str, default='coreml',
                   choices=['xnnpack', 'coreml', 'vulkan'],
                   help='selects model_<backend>.pte in --rec_model_dir')
    p.add_argument('--et_out_index', type=int, default=0,
                   help='index of the CTC logits in the flattened .pte outputs')
    p.add_argument('--warmup', action='store_true')
    return p.parse_args()


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    recognizer = TextRecognizerET(args)

    if args.warmup and image_file_list:
        wi, wf, _ = check_and_read(image_file_list[0])
        if not wf:
            wi = cv2.imread(image_file_list[0])
        if wi is not None:
            recognizer([wi])

    img_list, valid = [], []
    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info(f'error in loading image: {image_file}')
            continue
        valid.append(image_file)
        img_list.append(img)

    rec_res, elapse = recognizer(img_list)
    for path, res in zip(valid, rec_res):
        logger.info(f'result of {path}: {res}')
    logger.info(f'[{args.et_backend}] total predict time: {elapse:.3f}s '
                f'over {len(valid)} crops ({elapse / max(1, len(valid)) * 1000:.1f} ms/crop)')


if __name__ == '__main__':
    main(parse_args())
