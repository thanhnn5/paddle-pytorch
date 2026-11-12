"""Fast postprocess sweep on cached probability maps (CPU). Replicates
DBPostProcess.boxes_from_bitmap (quad, fast score) exactly but shares the
contour/score work across box_thresh and unclip values."""
import os, sys, pickle, itertools, numpy as np, cv2
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from common import *

_P = DBPostProcess(**FIXED_PP)


def sweep_image(args):
    prob, shape, polys, tags, TH, BT, UC = args
    prob = prob.astype(np.float32)
    src_h, src_w = shape[0], shape[1]
    H, W = prob.shape
    out = np.zeros((len(TH), len(BT), len(UC), 3), np.int64)
    for ti, th in enumerate(TH):
        bitmap = prob > th
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cand = []
        for c in contours[:_P.max_candidates]:
            pts, ss = _P.get_mini_boxes(c)
            if ss < _P.min_size:
                continue
            pts = np.array(pts)
            sc = _P.box_score_fast(prob, pts.reshape(-1, 2))
            if sc < BT[0]:
                continue
            cand.append((pts, sc))
        for ui, uc in enumerate(UC):
            bx = []
            for pts, sc in cand:
                box = _P.unclip(pts, uc).reshape(-1, 1, 2)
                box, ss = _P.get_mini_boxes(box)
                if ss < _P.min_size + 2:
                    continue
                box = np.array(box)
                box[:, 0] = np.clip(np.round(box[:, 0] / W * src_w), 0, src_w)
                box[:, 1] = np.clip(np.round(box[:, 1] / H * src_h), 0, src_h)
                bx.append((box.astype('int32'), sc))
            for bi, bt in enumerate(BT):
                out[ti, bi, ui] = score_image([b for b, s in bx if not (bt > s)], polys, tags)
    return out


def sweep(cache, TH, BT, UC, procs=4):
    r = cache if isinstance(cache, dict) else pickle.load(open(cache, 'rb'))
    BT = sorted(BT)
    jobs = [(r['probs'][i], r['shapes'][i], r['gts'][i][0], r['gts'][i][1], TH, BT, UC) for i in range(len(r['probs']))]
    with Pool(procs) as p:
        res = p.map(sweep_image, jobs, chunksize=8)
    return np.stack(res), r  # N x T x B x U x 3
