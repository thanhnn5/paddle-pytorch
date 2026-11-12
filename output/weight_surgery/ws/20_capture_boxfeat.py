"""Capture per-box pooled features for TP vs FP analysis.
For every candidate box (fixed thresh 0.2, box score >= 0.3) store: label (tp / fp_bg / fp_empty / other),
score, mean logit, and features mean-pooled over the box region at three depths:
  fuse (neck output, 96ch, 1/4), h1 (binarize conv_bn1+relu, 24ch, 1/4), h2 (conv_bn2+relu, 24ch, 1/2).
usage: 20_capture_boxfeat.py <label_file> <out.pkl> [ckpt]"""
import os, sys, pickle, numpy as np, torch, cv2
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image
lf, out = sys.argv[1:3]
ck = sys.argv[3] if len(sys.argv) > 3 else ORIG
model = make_model(load_sd(ck))
feats = {}
hb = model.head.binarize
h_ = [model.neck.register_forward_hook(lambda m, i, o: feats.__setitem__('fuse', o)),
      hb.conv_bn1.register_forward_hook(lambda m, i, o: feats.__setitem__('h1', F.relu(o))),
      hb.conv_bn2.register_forward_hook(lambda m, i, o: feats.__setitem__('h2', F.relu(o))),
      hb.conv3.register_forward_hook(lambda m, i, o: feats.__setitem__('logit', o))]
P = DBPostProcess(**dict(FIXED_PP, box_thresh=0.3))
rows = []; X = {'fuse': [], 'h1': [], 'h2': []}
dl = loader(lf, VAL_DIR)
with torch.no_grad():
    for rec in dl:
        i = rec['idx']
        prob = model(rec['image'][None].cuda())['res'][0, 0]
        pn = prob.cpu().numpy(); H, W = pn.shape; src_h, src_w = rec['shape'][:2]
        contours, _ = cv2.findContours(((pn > P.thresh) * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes, masks, scores = [], [], []
        for c in contours[:P.max_candidates]:
            pts, ss = P.get_mini_boxes(c)
            if ss < P.min_size: continue
            pts = np.array(pts); sc = P.box_score_fast(pn, pts.reshape(-1, 2))
            if P.box_thresh > sc: continue
            box = P.unclip(pts, P.unclip_ratio).reshape(-1, 1, 2)
            box, ss = P.get_mini_boxes(box)
            if ss < P.min_size + 2: continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / W * src_w), 0, src_w); box[:, 1] = np.clip(np.round(box[:, 1] / H * src_h), 0, src_h)
            m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, pts.reshape(1, -1, 2).astype('int32'), 1)
            boxes.append(box.astype('int32')); masks.append(m); scores.append(sc)
        if not boxes: continue
        polys, tags = rec['polys'], rec['tags']
        dets, gts, _ = analyse_image(boxes, polys, tags)
        cat = {d['d']: d['cat'] for d in dets}
        M = torch.from_numpy(np.stack(masks)).float().cuda()[:, None]      # K x 1 x H x W
        lg = (feats['logit'][0, 0][None, None] * M).sum((2, 3)) / M.sum((2, 3))
        pooled = {}
        for k, s in [('fuse', 4), ('h1', 4), ('h2', 2)]:
            Ms = F.avg_pool2d(M, s)                                        # K x 1 x h x w
            f = feats[k][0][None]                                          # 1 x C x h x w
            pooled[k] = ((f * Ms).sum((2, 3)) / Ms.sum((2, 3)).clamp_min(1e-6)).cpu().numpy()
        for d in range(len(boxes)):
            c = cat.get(d, 'tp')
            lab = {'tp': 'tp', 'fp_background': 'fp_bg', 'fp_empty_image': 'fp_empty'}.get(c, 'other')
            rows.append(dict(img=i, d=d, label=lab, cat=c, score=float(scores[d]), logit=float(lg[d, 0]), area=float(masks[d].sum()),
                             box=boxes[d].tolist()))
            for k in X: X[k].append(pooled[k][d])
names = [n for n, _ in dl.dataset.items]
pickle.dump(dict(rows=rows, X={k: np.stack(v) for k, v in X.items()}, names=names), open(out, 'wb'))
import collections
print(os.path.basename(lf), len(rows), collections.Counter(r['label'] for r in rows), 'score>=0.45:', collections.Counter(r['label'] for r in rows if r['score'] >= 0.45))
