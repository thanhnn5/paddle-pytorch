"""Detection inference with an ExecuTorch .pte (xnnpack / coreml).

Mirrors tools/infer/mnn/predict_det.py: config-driven pre/post-processing read
from the `config.yaml` saved next to the .pte, with the same CLI overrides.

Two ExecuTorch-specific gotchas, both handled here:

  1. Input dim-order. The runtime reads the input buffer according to the
     dim-order the program was EXPORTED with (a C-contiguous torch.randn dummy),
     and does NOT honour arbitrary tensor strides. torchocr preprocessing
     returns a channels_last-strided ndarray (ToCHWImage's transpose + numpy
     astype(order='K')), so feeding it verbatim silently collapses the prob map
     to ~0.14 max. We force C-contiguous NCHW with np.ascontiguousarray.

  2. Output selection. The det head returns {'res', 'cbn_maps'} in eval mode;
     ExecuTorch flattens this to a tuple where index 0 is the 'res' prob map
     DBPostProcess needs. Override with --et_out_index if a model differs.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

import cv2
import json
import argparse
import numpy as np
import time

import torch

from torchocr import Config
from torchocr.postprocess import build_post_process
from torchocr.data.imaug import create_operators, transform
from torchocr.utils.logging import get_logger
from torchocr.utils.visual import draw_det
from torchocr.utils.utility import get_image_file_list, check_and_read

logger = get_logger()


class ExecuTorchEngine:
    """Thin wrapper over executorch.runtime, NCHW float32 in, list[ndarray] out."""

    def __init__(self, pte_path):
        if not os.path.exists(pte_path):
            raise Exception(f'{pte_path} is not exists')
        from executorch.runtime import Runtime
        self.method = Runtime.get().load_program(pte_path).load_method('forward')

    def run(self, image_numpy):
        # ascontiguousarray: the runtime ignores strides and reads the raw
        # buffer per the export dim-order (C-contiguous). Skipping this collapses
        # the prob map. See module docstring.
        x = torch.from_numpy(np.ascontiguousarray(image_numpy.astype(np.float32)))
        out = self.method.execute([x])
        outs = out if isinstance(out, (list, tuple)) else [out]
        return [t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)
                for t in outs]


class TextDetectorET:
    def __init__(self, args):
        if args.det_model_dir is None or not os.path.exists(args.det_model_dir):
            raise Exception(f'args.det_model_dir is set to {args.det_model_dir}, but it is not exists')

        pte_path = os.path.join(args.det_model_dir, f'model_{args.et_backend}.pte')
        config_path = os.path.join(args.det_model_dir, 'config.yaml')

        self.engine = ExecuTorchEngine(pte_path)
        self.args = args
        self.out_index = args.et_out_index
        cfg = Config(config_path).cfg

        # Strip DecodeImage (caller passes an ndarray) and BGR→RGB ourselves to
        # match the training img_mode — identical to the MNN runner.
        self.img_mode = (cfg.get('Transforms', [{}])[0].get('DecodeImage') or {}).get('img_mode', 'RGB')
        resize = {'DetResizeForTest': {'limit_side_len': args.det_limit_side_len,
                                       'limit_type':     args.det_limit_type}}
        pre_process_list = [
            resize if 'DetResizeForTest' in op else op
            for op in cfg.get('Transforms', []) if 'DecodeImage' not in op
        ]
        self.preprocess_op = create_operators(pre_process_list)

        post_overrides = {k: v for k, v in {
            'thresh':       args.det_db_thresh,
            'box_thresh':   args.det_db_box_thresh,
            'unclip_ratio': args.det_db_unclip_ratio,
        }.items() if v is not None}
        self.postprocess_op = build_post_process({**cfg['PostProcess'], **post_overrides})

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
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = {'image': img}
        st = time.time()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)

        preds = self.engine.run(img)
        prob_map = preds[self.out_index]

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
    p.add_argument('--output', type=str, default='./et_det_out')
    p.add_argument('--det_algorithm', type=str, default='DB')
    p.add_argument('--det_box_type', type=str, default='quad')
    p.add_argument('--det_limit_side_len', type=float, default=1280)
    p.add_argument('--det_limit_type', type=str, default='max')
    p.add_argument('--det_db_thresh',       type=float, default=None)
    p.add_argument('--det_db_box_thresh',   type=float, default=None)
    p.add_argument('--det_db_unclip_ratio', type=float, default=None)
    p.add_argument('--et_backend', type=str, default='coreml',
                   choices=['xnnpack', 'coreml', 'vulkan'],
                   help='selects model_<backend>.pte in --det_model_dir')
    p.add_argument('--et_out_index', type=int, default=0,
                   help="index of the 'res' prob map in the flattened .pte outputs")
    p.add_argument('--warmup', action='store_true')
    return p.parse_args()


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    detector = TextDetectorET(args)

    os.makedirs(args.output, exist_ok=True)

    # Warmup on the FIRST real image — the .pte is exported at a fixed shape and
    # the runtime cannot resize a static input (a synthetic 640x640 dummy would
    # raise "Attempted to resize a static tensor"). Running the real image also
    # primes CoreML's on-disk model compilation cache.
    if args.warmup and image_file_list:
        wi, wf, _ = check_and_read(image_file_list[0])
        if not wf:
            wi = cv2.imread(image_file_list[0])
        if wi is not None:
            for _ in range(2):
                detector(wi)

    total = 0.0
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
            total += elapse

            dt_boxes_json = [{'transcription': '', 'points': np.array(b).tolist()} for b in dt_boxes]
            out_str = f'{image_file}\t{json.dumps(dt_boxes_json)}'
            fout.write(out_str + '\n')
            logger.info(out_str)
            logger.info(f'[{args.et_backend}] {len(dt_boxes)} boxes, predict time {os.path.basename(image_file)}: {elapse:.3f}s')

            vis = draw_det(dt_boxes, img)
            save_path = os.path.join(args.output, f'inference_det_{os.path.basename(image_file)}')
            cv2.imwrite(save_path, vis)
    logger.info(f'[{args.et_backend}] mean predict time over {len(image_file_list)} imgs: '
                f'{total / max(1, len(image_file_list)):.3f}s')


if __name__ == '__main__':
    main(parse_args())
