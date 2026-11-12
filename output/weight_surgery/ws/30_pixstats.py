"""Pixel-level second-moment statistics of h2 (input of head.binarize.conv3) on train images, by region and
logit bin, for closed-form least-squares edits of the final layer.
Regions: 0 FP box (det with no GT overlap), 1 TP box, 2 GT not matched, 3 elsewhere (background). 'other' dets ignored.
Logit bins: edges [-6,-3,0,3,6]. Stats per stratum: sum of [h,1][h,1]^T (25x25), count, and sum of logit*[h,1].
usage: 30_pixstats.py <label_file> <out.pkl> [ckpt]"""
import os, sys, pickle, numpy as np, torch, cv2
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image
lf, out = sys.argv[1:3]; ck = sys.argv[3] if len(sys.argv) > 3 else ORIG
model = make_model(load_sd(ck)); hb = model.head.binarize; feats = {}
hb.conv_bn2.register_forward_hook(lambda m, i, o: feats.__setitem__('h2', F.relu(o)))
hb.conv3.register_forward_hook(lambda m, i, o: feats.__setitem__('logit', o))
P = DBPostProcess(**dict(FIXED_PP, box_thresh=0.3))
EDGES = torch.tensor([-6., -3., 0., 3., 6.]).cuda()
C = torch.zeros(4, 6, 25, 25, dtype=torch.float64).cuda(); Nn = torch.zeros(4, 6, dtype=torch.float64).cuda()
dl = loader(lf, VAL_DIR)
with torch.no_grad():
    for rec in dl:
        prob = model(rec['image'][None].cuda())['res'][0, 0]
        pn = prob.cpu().numpy(); H, W = pn.shape; src_h, src_w = rec['shape'][:2]
        contours, _ = cv2.findContours(((pn > P.thresh) * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes, minis = [], []
        for c in contours[:P.max_candidates]:
            pts, ss = P.get_mini_boxes(c)
            if ss < P.min_size: continue
            pts = np.array(pts); sc = P.box_score_fast(pn, pts.reshape(-1, 2))
            if P.box_thresh > sc: continue
            box = P.unclip(pts, P.unclip_ratio).reshape(-1, 1, 2); box, ss = P.get_mini_boxes(box)
            if ss < P.min_size + 2: continue
            box = np.array(box); box[:, 0] = np.clip(np.round(box[:, 0] / W * src_w), 0, src_w); box[:, 1] = np.clip(np.round(box[:, 1] / H * src_h), 0, src_h)
            boxes.append(box.astype('int32')); minis.append(pts)
        polys, tags = rec['polys'], rec['tags']
        dets, gts, _ = analyse_image(boxes, polys, tags) if boxes else ([], [dict(g=g) for g in range(len(polys)) if not tags[g]], [])
        cat = {d['d']: d['cat'] for d in dets}
        reg = np.full((H, W), 3, np.uint8); ign = np.zeros((H, W), np.uint8)
        sx, sy = W / src_w, H / src_h
        for g, (p, t) in enumerate(zip(polys, tags)):
            q = (np.asarray(p, np.float64) * [sx, sy]).astype(np.int32)
            if t: cv2.fillPoly(ign, [q], 1)
        for x in gts:  # unmatched GT
            q = (np.asarray(polys[x['g']], np.float64) * [sx, sy]).astype(np.int32); cv2.fillPoly(reg, [q], 2)
        for d, pts in enumerate(minis):
            c = cat.get(d, 'tp'); big = P.unclip(pts, 1.4).reshape(-1, 2).astype(np.int32)
            if c == 'tp': cv2.fillPoly(reg, [big], 1)
            elif c in ('fp_background', 'fp_empty_image'): cv2.fillPoly(reg, [big], 0)
            else: cv2.fillPoly(ign, [big], 1)
        h2 = feats['h2'][0]; lg = F.avg_pool2d(feats['logit'], 2)[0, 0]               # 24 x h x w ; h x w
        regt = torch.from_numpy(reg).cuda()[::2, ::2].long(); keep = torch.from_numpy(ign).cuda()[::2, ::2] == 0
        X = torch.cat([h2, torch.ones_like(h2[:1])], 0).flatten(1).double()            # 25 x n
        binid = torch.bucketize(lg.flatten(), EDGES); st = (regt.flatten() * 6 + binid)[keep.flatten()]
        X = X[:, keep.flatten()]
        for s in st.unique().tolist():
            Xs = X[:, st == s]; C[s // 6, s % 6] += Xs @ Xs.T; Nn[s // 6, s % 6] += Xs.shape[1]
pickle.dump(dict(C=C.cpu().numpy(), N=Nn.cpu().numpy(), edges=EDGES.cpu().numpy()), open(out, 'wb'))
np.set_printoptions(suppress=True, linewidth=200)
print('pixel counts (rows: FP box, TP box, missed GT, background; cols: logit bins <-6,-6..-3,-3..0,0..3,3..6,>6)'); print(Nn.cpu().numpy().astype(int))
