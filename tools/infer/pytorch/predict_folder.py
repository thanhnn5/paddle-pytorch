# -*- coding: utf-8 -*-
"""Batch OCR over a folder of images.

Reuses TextSystem from predict_system.py. For each image writes a line
`path<TAB>json_results` to the output file (results may be an empty list).

Optionally, with --mask_file, loads a prior detection result (TSV
`path<TAB>json` with "points" polygons) and blacks out each detected
polygon before OCR — e.g. to suppress package regions so they don't
interfere with the address-numeral recognition step.
"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..', '..')))

import json
import time
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from torchocr.utils.utility import get_image_file_list
from torchocr.utils.logging import get_logger
from tools.infer.pytorch.predict_system import TextSystem, read_image, add_system_args

logger = get_logger()


def load_mask_boxes(mask_path):
    """path -> list of polygons (each an Nx2 point list) from a detection file.

    Expects the TSV detection-result format: `path<TAB>json`, where json is a
    list of {"points": [[x,y], ...], ...} (an empty list means no detections).
    """
    by_path = {}
    with open(mask_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            path, payload = line.split('\t', 1)
            dets = json.loads(payload)
            by_path[path] = [d['points'] for d in dets if 'points' in d]
    return by_path


def mask_boxes(img, polygons):
    """Fill each detected polygon with black (0,0,0). In-place."""
    for points in polygons:
        poly = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(img, [poly], (0, 0, 0))
    return img


def parse_args():
    p = argparse.ArgumentParser(description='PyTorch det->rec OCR — batch folder run')
    p.add_argument('--image_dir', type=str, required=True)
    p.add_argument('--output', type=str, default='./output/system_results.txt')
    p.add_argument('--mask_file', '--mask_jsonl', dest='mask_file', type=str, default=None,
                   help='detection result file (TSV `path<TAB>json` with "points"); '
                        'black out each detected polygon before OCR')
    p.add_argument('--warmup', action='store_true', default=True,
                   help='prime the accelerator on the first image (default on)')
    p.add_argument('--no_warmup', dest='warmup', action='store_false')
    add_system_args(p)
    return p.parse_args()


def main(args):
    system = TextSystem(args)
    image_file_list = get_image_file_list(args.image_dir)
    logger.info(f'{len(image_file_list)} images found in {args.image_dir}')

    mask_by_path = {}
    if args.mask_file:
        mask_by_path = load_mask_boxes(args.mask_file)
        n_boxes = sum(len(v) for v in mask_by_path.values())
        logger.info(f'masking {n_boxes} polygons over {len(mask_by_path)} images '
                    f'from {args.mask_file}')

    # First MPS/CUDA call pays kernel-compile + lazy-init cost. Prime it on the
    # first real image so the timed run reflects steady-state throughput.
    if args.warmup and image_file_list:
        warm = read_image(image_file_list[0])
        if warm is not None:
            for _ in range(2):
                system(warm)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    n_boxes_total = 0
    start = time.time()
    with open(args.output, 'w', encoding='utf-8') as fout:
        pbar = tqdm(image_file_list, desc='OCR', unit='img')
        for image_file in pbar:
            img = read_image(image_file)
            if img is None:
                pbar.write(f'error loading image: {image_file}')
                continue
            if args.mask_file:
                mask_boxes(img, mask_by_path.get(image_file, []))
            results = system(img)
            n_boxes_total += len(results)
            fout.write(f'{image_file}\t{json.dumps(results, ensure_ascii=False)}\n')
            fout.flush()
            pbar.set_postfix(boxes=n_boxes_total)

    elapse = time.time() - start
    logger.info(f'done. {n_boxes_total} boxes over {len(image_file_list)} images '
                f'in {elapse:.1f}s ({elapse/max(1,len(image_file_list))*1000:.0f}ms/img) '
                f'-> {args.output}')


if __name__ == '__main__':
    main(parse_args())
