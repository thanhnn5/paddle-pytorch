import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

import cv2
import argparse
import math
import numpy as np
import time

from torchocr import Config
from torchocr.postprocess import build_post_process
from torchocr.utils.logging import get_logger
from torchocr.utils.utility import get_image_file_list, check_and_read
from tools.infer.mnn.engine import MNNEngine, BACKEND_MAP, PRECISION_MAP

logger = get_logger()


class TextRecognizerMNN:
    """
    MNN rec runner for the static [1, 3, 48, 320] export of PP-OCRv5 mobile rec.

    The exported model has a fixed input shape, so unlike the ONNX path we
    don't bucket by max_wh_ratio — every crop is letterboxed into 48x320.
    """

    def __init__(self, args):
        if args.rec_model_dir is None or not os.path.exists(args.rec_model_dir):
            raise Exception(f'args.rec_model_dir is set to {args.rec_model_dir}, but it is not exists')

        mnn_path = os.path.join(args.rec_model_dir, 'model.mnn')
        config_path = os.path.join(args.rec_model_dir, 'config.yaml')

        self.engine = MNNEngine(
            mnn_path,
            backend=args.mnn_backend,
            precision=args.mnn_precision,
            input_names=['input'],
            output_names=['output'],
        )

        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(',')]
        cfg = Config(config_path).cfg
        self.postprocess_op = build_post_process(cfg['PostProcess'])

        # Honor the img_mode the model was trained with. cv2.imread / det crops
        # are BGR; flip to RGB if the rec was trained that way. PP-OCRv5 mobile
        # rec is BGR (matches default), so this is a no-op for it.
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
        # Static-shape export → batch_size=1, one forward per image.
        for i, img in enumerate(img_list):
            norm = self.resize_norm_img(img)[np.newaxis, :]
            preds = self.engine.run(norm)[0]
            out = self.postprocess_op({'res': preds})
            rec_res[i] = out[0]
        return rec_res, time.time() - st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rec_model_dir', type=str, required=True)
    p.add_argument('--image_dir', type=str, required=True)
    p.add_argument('--rec_image_shape', type=str, default='3,48,320')
    p.add_argument('--mnn_backend', type=str, default='metal',
                   choices=sorted(BACKEND_MAP.keys()))
    p.add_argument('--mnn_precision', type=str, default='low',
                   choices=sorted(PRECISION_MAP.keys()))
    p.add_argument('--warmup', action='store_true')
    return p.parse_args()


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    recognizer = TextRecognizerMNN(args)

    if args.warmup:
        dummy = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        recognizer([dummy])

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
    logger.info(f'total predict time: {elapse:.3f}s')


if __name__ == '__main__':
    main(parse_args())
