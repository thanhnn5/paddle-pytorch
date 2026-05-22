import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

import cv2
import copy
import json
import time
import argparse
import numpy as np
from PIL import Image

from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
from tools.infer.mnn.predict_det import TextDetectorMNN
from tools.infer.mnn.predict_rec import TextRecognizerMNN
from tools.infer.mnn.engine import BACKEND_MAP, PRECISION_MAP
from torchocr.utils.utility import get_image_file_list, check_and_read
from torchocr.utils.logging import get_logger
from torchocr.utils.visual import draw_system

logger = get_logger()


def sorted_boxes(dt_boxes):
    """Top-to-bottom, left-to-right ordering used in tools/infer/onnx/predict_system.py."""
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


class TextSystemMNN(object):
    def __init__(self, args):
        self.text_detector = TextDetectorMNN(args)
        self.text_recognizer = TextRecognizerMNN(args)
        self.drop_score = args.drop_score
        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        for bno, crop in enumerate(img_crop_list):
            cv2.imwrite(os.path.join(output_dir, f'mg_crop_{bno + self.crop_image_res_index}.jpg'), crop)
            logger.debug(f'{bno}, {rec_res[bno]}')
        self.crop_image_res_index += len(img_crop_list)

    def __call__(self, img):
        time_dict = {'det': 0, 'rec': 0, 'all': 0}
        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, det_elapse = self.text_detector(img)
        time_dict['det'] = det_elapse

        if dt_boxes is None or len(dt_boxes) == 0:
            time_dict['all'] = time.time() - start
            return None, None, time_dict

        dt_boxes = sorted_boxes(dt_boxes)
        img_crop_list = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            if self.args.det_box_type == 'quad':
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = rec_elapse

        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)

        filter_boxes, filter_rec_res = [], []
        for box, (text, score) in zip(dt_boxes, rec_res):
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append((text, score))

        time_dict['all'] = time.time() - start
        return filter_boxes, filter_rec_res, time_dict


def parse_args():
    p = argparse.ArgumentParser()
    # Inputs / outputs
    p.add_argument('--image_dir', type=str, required=True)
    p.add_argument('--output', type=str, default='./mnn_system_out')
    p.add_argument('--save_crop_res', action='store_true')
    p.add_argument('--crop_res_save_dir', type=str, default='./mnn_system_out/crops')
    p.add_argument('--drop_score', type=float, default=0.5)
    p.add_argument('--vis_font_path', type=str, default='./doc/fonts/simfang.ttf')
    p.add_argument('--no_visualize', action='store_true')
    # Det
    p.add_argument('--det_model_dir', type=str, required=True)
    p.add_argument('--det_algorithm', type=str, default='DB')
    p.add_argument('--det_box_type', type=str, default='quad')
    p.add_argument('--det_limit_side_len', type=float, default=1280)
    p.add_argument('--det_limit_type', type=str, default='max')
    p.add_argument('--det_db_thresh',       type=float, default=None)
    p.add_argument('--det_db_box_thresh',   type=float, default=None)
    p.add_argument('--det_db_unclip_ratio', type=float, default=None)
    # Rec
    p.add_argument('--rec_model_dir', type=str, required=True)
    p.add_argument('--rec_image_shape', type=str, default='3,48,320')
    # MNN runtime — shared by det + rec for simplicity. Override per-stage by
    # editing the args.Namespace after parse if you need different settings
    # (e.g. det on metal-high, rec on metal-low).
    p.add_argument('--mnn_backend', type=str, default='metal',
                   choices=sorted(BACKEND_MAP.keys()))
    p.add_argument('--mnn_precision', type=str, default='low',
                   choices=sorted(PRECISION_MAP.keys()))
    p.add_argument('--warmup', action='store_true')
    return p.parse_args()


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    sys_pipeline = TextSystemMNN(args)
    os.makedirs(args.output, exist_ok=True)

    if args.warmup:
        dummy = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for _ in range(2):
            sys_pipeline(dummy)

    total_time = 0.0
    out_path = os.path.join(args.output, 'inference_system.txt')
    with open(out_path, 'w') as fout:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.info(f'error in loading image: {image_file}')
                continue

            tic = time.time()
            dt_boxes, rec_res, time_dict = sys_pipeline(img)
            elapse = time.time() - tic
            total_time += elapse

            if dt_boxes is None:
                logger.info(f'{image_file}: no detections (elapse={elapse:.3f}s)')
                fout.write(f'{image_file}\t[]\n')
                continue

            res = [{
                'transcription': rec_res[i][0],
                'confidence': float(rec_res[i][1]),
                'points': np.array(dt_boxes[i]).astype(np.int32).tolist(),
            } for i in range(len(dt_boxes))]
            fout.write(f'{image_file}\t{json.dumps(res, ensure_ascii=False)}\n')
            logger.info(f'{image_file}: {len(dt_boxes)} boxes  '
                        f'det={time_dict["det"]*1000:.1f}ms rec={time_dict["rec"]*1000:.1f}ms '
                        f'all={time_dict["all"]*1000:.1f}ms')

            if not args.no_visualize:
                image_rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                txts = [r[0] for r in rec_res]
                scores = [r[1] for r in rec_res]
                draw_img = draw_system(image_rgb, dt_boxes, txts, scores,
                                       drop_score=args.drop_score,
                                       font_path=args.vis_font_path)
                save_path = os.path.join(args.output, f'inference_system_{os.path.basename(image_file)}')
                cv2.imwrite(save_path, draw_img[:, :, ::-1])

    logger.info(f'wrote {out_path}  total={total_time:.3f}s  avg={total_time/max(1,len(image_file_list))*1000:.1f}ms/image')


if __name__ == '__main__':
    main(parse_args())
