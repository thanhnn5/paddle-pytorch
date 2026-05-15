import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

import cv2
import json
import argparse
import numpy as np
import time

from torchocr import Config
from torchocr.postprocess import build_post_process
from torchocr.data.imaug import create_operators, transform
from torchocr.utils.logging import get_logger
from torchocr.utils.visual import draw_det
from torchocr.utils.utility import get_image_file_list, check_and_read
from tools.infer.mnn.engine import MNNEngine, BACKEND_MAP, PRECISION_MAP

logger = get_logger()


class TextDetectorMNN:
    def __init__(self, args):
        if args.det_model_dir is None or not os.path.exists(args.det_model_dir):
            raise Exception(f'args.det_model_dir is set to {args.det_model_dir}, but it is not exists')

        mnn_path = os.path.join(args.det_model_dir, 'model.mnn')
        config_path = os.path.join(args.det_model_dir, 'config.yaml')

        # The det ONNX exposes two outputs (`output`, `sigmoid_1`); we only need the
        # probability map, which is `output`.
        self.engine = MNNEngine(
            mnn_path,
            backend=args.mnn_backend,
            precision=args.mnn_precision,
            input_names=['input'],
            output_names=['output'],
        )

        self.args = args
        self.det_algorithm = args.det_algorithm
        cfg = Config(config_path).cfg

        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(cfg['PostProcess'])

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        return np.array(dt_boxes_new)

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        return np.array(dt_boxes_new)

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        st = time.time()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0).copy()
        shape_list = np.expand_dims(shape_list, axis=0)

        preds = self.engine.run(img)
        prob_map = preds[0]

        post_result = self.postprocess_op({'res': prob_map}, [-1, shape_list])
        dt_boxes = post_result[0]['points']

        if self.args.det_box_type == 'poly':
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes, time.time() - st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--det_model_dir', type=str, required=True)
    p.add_argument('--image_dir', type=str, required=True)
    p.add_argument('--output', type=str, default='./mnn_det_out')
    p.add_argument('--det_algorithm', type=str, default='DB')
    p.add_argument('--det_box_type', type=str, default='quad')
    p.add_argument('--det_limit_side_len', type=float, default=1280)
    p.add_argument('--det_limit_type', type=str, default='max')
    p.add_argument('--mnn_backend', type=str, default='metal',
                   choices=sorted(BACKEND_MAP.keys()),
                   help='MNN forward type (see BACKEND_MAP in mnn_engine.py)')
    p.add_argument('--mnn_precision', type=str, default='low',
                   choices=sorted(PRECISION_MAP.keys()),
                   help='MNN precision mode (see PRECISION_MAP in mnn_engine.py)')
    p.add_argument('--warmup', action='store_true')
    return p.parse_args()


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    detector = TextDetectorMNN(args)

    os.makedirs(args.output, exist_ok=True)

    if args.warmup:
        dummy = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for _ in range(2):
            detector(dummy)

    with open(os.path.join(args.output, 'inference_det.txt'), 'w') as fout:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.info(f'error in loading image: {image_file}')
                continue

            tic = time.time()
            dt_boxes, _ = detector(img)
            elapse = time.time() - tic

            dt_boxes_json = [{'transcription': '', 'points': np.array(b).tolist()} for b in dt_boxes]
            out_str = f'{image_file}\t{json.dumps(dt_boxes_json)}'
            fout.write(out_str + '\n')
            logger.info(out_str)
            logger.info(f'predict time {image_file}: {elapse:.3f}s')

            vis = draw_det(dt_boxes, img)
            save_path = os.path.join(args.output, f'inference_det_{os.path.basename(image_file)}')
            cv2.imwrite(save_path, vis)


if __name__ == '__main__':
    main(parse_args())
