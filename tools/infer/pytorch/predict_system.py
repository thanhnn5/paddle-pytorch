# -*- coding: utf-8 -*-
"""PyTorch det->rec OCR system — single-image inference.

Chains a detection model (config + .pth) and a recognition model
(config + .pth). Run directly to OCR a single image and print the
raw results. For batch/folder runs use predict_folder.py, which
imports the TextSystem class defined here.
"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..', '..')))

import cv2
import copy
import json
import argparse
import numpy as np
import torch

from torchocr.data import create_operators, transform
from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils.ckpt import load_ckpt
from torchocr.utils.utility import check_and_read
from torchocr.utils.logging import get_logger
from tools.utility import update_rec_head_out_channels
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
from torchocr import Config

logger = get_logger()


def build_det_process(cfg, limit_side_len=1280, limit_type='max'):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'DetResizeForTest':
            op[op_name] = {'limit_side_len': limit_side_len, 'limit_type': limit_type}
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)
    return transforms


def build_rec_process(cfg):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    return transforms


def sorted_boxes(dt_boxes):
    """Top-to-bottom, left-to-right ordering."""
    num_boxes = dt_boxes.shape[0]
    _boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(_boxes)
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return _boxes


class TextDetector(object):
    def __init__(self, cfg, device, limit_side_len=1280, limit_type='max'):
        self.device = device
        self.post_process_class = build_post_process(cfg['PostProcess'])
        self.model = build_model(cfg['Architecture'])
        load_ckpt(self.model, cfg)
        self.model.to(device)
        self.model.eval()
        self.ops = create_operators(
            build_det_process(cfg, limit_side_len, limit_type), cfg['Global'])

    def __call__(self, img):
        # img: BGR ndarray (as read by cv2)
        data = {'image': cv2.imencode('.png', img)[1].tobytes()}
        batch = transform(data, self.ops)
        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = torch.from_numpy(images).to(self.device)
        with torch.no_grad():
            preds = self.model(images)
        post_result = self.post_process_class(preds, [-1, shape_list])
        return post_result[0]['points']


class TextRecognizer(object):
    def __init__(self, cfg, device):
        self.device = device
        self.algorithm = cfg['Architecture']['algorithm']
        self.post_process_class = build_post_process(cfg['PostProcess'])
        update_rec_head_out_channels(cfg, self.post_process_class)
        self.model = build_model(cfg['Architecture'])
        load_ckpt(self.model, cfg)
        self.model.to(device)
        self.model.eval()
        cfg['Global']['infer_mode'] = True
        self.ops = create_operators(build_rec_process(cfg), cfg['Global'])

    def __call__(self, img_crop):
        # img_crop: BGR ndarray of a single text region
        data = {'image': cv2.imencode('.png', img_crop)[1].tobytes()}
        batch = transform(data, self.ops)
        images = np.expand_dims(batch[0], axis=0)
        images = torch.from_numpy(images).to(self.device)
        with torch.no_grad():
            preds = self.model(images, None)
        post_result = self.post_process_class(preds)
        # SVTR_LCNet/CTC -> list of (text, score)
        text, score = post_result[0][0], float(post_result[0][1])
        return text, score


def pick_device(prefer='auto'):
    if prefer != 'auto':
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TextSystem(object):
    """det -> crop -> rec over a single BGR image. Returns a list of
    {transcription, confidence, points} dicts, sorted top-to-bottom."""

    def __init__(self, args):
        self.device = pick_device(args.device)
        logger.info(f'device: {self.device}')
        det_cfg = Config(args.det_config).cfg
        rec_cfg = Config(args.rec_config).cfg
        det_cfg['Global']['checkpoints'] = args.det_model
        rec_cfg['Global']['checkpoints'] = args.rec_model
        self.detector = TextDetector(det_cfg, self.device,
                                     limit_side_len=args.det_limit_side_len,
                                     limit_type=args.det_limit_type)
        self.recognizer = TextRecognizer(rec_cfg, self.device)
        self.det_box_type = args.det_box_type
        self.drop_score = args.drop_score

    def __call__(self, img):
        if img is None:
            return None
        ori_im = img.copy()
        dt_boxes = self.detector(img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return []
        dt_boxes = sorted_boxes(np.array(dt_boxes))
        results = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(np.array(box, dtype=np.float32))
            if self.det_box_type == 'quad':
                crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                crop = get_minarea_rect_crop(ori_im, tmp_box)
            text, score = self.recognizer(crop)
            if score < self.drop_score:
                continue
            results.append({
                'transcription': text,
                'confidence': score,
                'points': np.array(box).astype(np.int32).tolist(),
            })
        return results


def read_image(image_file):
    img, flag, _ = check_and_read(image_file)
    if not flag:
        img = cv2.imread(image_file)
    return img


def add_system_args(p):
    """Shared det/rec/device args (reused by predict_folder.py)."""
    p.add_argument('--det_config', type=str,
                   default='configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml')
    p.add_argument('--det_model', type=str,
                   default='weights/dinov3/convnext_det_unfreeze.pth')
    p.add_argument('--rec_config', type=str,
                   default='configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml')
    p.add_argument('--rec_model', type=str, default='weights/mob_rec_2.pth')
    p.add_argument('--det_box_type', type=str, default='quad', choices=['quad', 'poly'])
    p.add_argument('--det_limit_side_len', type=float, default=1280)
    p.add_argument('--det_limit_type', type=str, default='max', choices=['max', 'min'])
    p.add_argument('--device', type=str, default='auto',
                   choices=['auto', 'mps', 'cuda', 'cpu'])
    p.add_argument('--drop_score', type=float, default=0.5)
    return p


def parse_args():
    p = argparse.ArgumentParser(description='PyTorch det->rec OCR — single image')
    p.add_argument('--image', type=str, required=True, help='path to a single image')
    add_system_args(p)
    return p.parse_args()


def main(args):
    system = TextSystem(args)
    img = read_image(args.image)
    if img is None:
        logger.error(f'could not load image: {args.image}')
        return
    results = system(img)
    raw_text = ' '.join(r['transcription'] for r in results) if results else ''
    logger.info(f'{args.image}: {len(results)} boxes | {raw_text}')
    print(json.dumps({'image': args.image, 'results': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(parse_args())
