# -*- coding: utf-8 -*-
"""Batch OCR over a folder of images.

Reuses TextSystem from predict_system.py. For each image writes a line
`path<TAB>json_results` to the output file (results may be an empty list).
"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..', '..')))

import json
import time
import argparse
from tqdm import tqdm

from torchocr.utils.utility import get_image_file_list
from torchocr.utils.logging import get_logger
from tools.infer.pytorch.predict_system import TextSystem, read_image, add_system_args

logger = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description='PyTorch det->rec OCR — batch folder run')
    p.add_argument('--image_dir', type=str, required=True)
    p.add_argument('--output', type=str, default='./output/system_results.txt')
    p.add_argument('--warmup', action='store_true', default=True,
                   help='prime the accelerator on the first image (default on)')
    p.add_argument('--no_warmup', dest='warmup', action='store_false')
    add_system_args(p)
    return p.parse_args()


def main(args):
    system = TextSystem(args)
    image_file_list = get_image_file_list(args.image_dir)
    logger.info(f'{len(image_file_list)} images found in {args.image_dir}')

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
