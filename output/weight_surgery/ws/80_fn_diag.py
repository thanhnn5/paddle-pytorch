"""Recall diagnosis: why is each unmatched GT unmatched at the fixed postprocess?
usage: 80_fn_diag.py <logits_cache.pkl> <tag> [data_dir]
For every unmatched care-GT: probability statistics inside the GT polygon, and the fate of every thresh-0.2
contour that overlaps it (dropped by min_size, dropped by box_thresh, kept but IoU<=0.5)."""
import os, sys, pickle, json, collections, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image, crop_sheet, poly
cache, tag = sys.argv[1:3]; dd = sys.argv[3] if len(sys.argv) > 3 else VAL_DIR
r = pickle.load(open(cache, 'rb'))
P = DBPostProcess(**FIXED_PP)
recs = []; tp_stats = []
for i in range(len(r['names'])):
    polys, tags = r['gts'][i]
    if len(polys) == 0: continue
    lg = r['logits'][i].astype(np.float32); prob = 1 / (1 + np.exp(-lg)); H, W = prob.shape; src_h, src_w = r['shapes'][i][:2]
    boxes = postprocess(prob, r['shapes'][i], P)
    dets, gts, _ = analyse_image(boxes, polys, tags)
    un = {g['g']: g for g in gts}
    # all contours with their fate
    contours, _ = cv2.findContours(((prob > P.thresh) * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours[:P.max_candidates]:
        pts, ss = P.get_mini_boxes(c); pts = np.array(pts)
        if ss < P.min_size: cands.append((pts, None, 'min_size', ss)); continue
        sc = P.box_score_fast(prob, pts.reshape(-1, 2))
        box = P.unclip(pts, P.unclip_ratio).reshape(-1, 1, 2); box, ss2 = P.get_mini_boxes(box); box = np.array(box)
        fate = 'score' if P.box_thresh > sc else ('min_size2' if ss2 < P.min_size + 2 else 'kept')
        box[:, 0] = np.clip(np.round(box[:, 0] / W * src_w), 0, src_w); box[:, 1] = np.clip(np.round(box[:, 1] / H * src_h), 0, src_h)
        cands.append((box, sc, fate, ss))
    for g, (p, t) in enumerate(zip(polys, tags)):
        if t: continue
        pm = np.asarray(p, np.float64) * [W / src_w, H / src_h]
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [np.round(pm).astype(np.int32)], 1); m = mask > 0
        if m.sum() == 0: continue
        rect = cv2.minAreaRect(pm.astype(np.float32)); short = min(rect[1]); longs = max(rect[1])
        st = dict(i=i, g=g, short=float(short), long=float(longs), pmax=float(prob[m].max()), frac=float((prob[m] > P.thresh).mean()),
                  lmean=float(lg[m].mean()), lmax=float(lg[m].max()), src_short=float(short * src_h / H))
        if g not in un: tp_stats.append(st); continue
        st['cat'] = un[g]['cat']; st['best_iou'] = un[g].get('best_iou', 0.)
        G = poly(p); ov = []
        for box, sc, fate, ss in cands:
            D = poly(box)
            it = G.intersection(D).area
            if it > 0.1 * min(G.area, max(D.area, 1e-9)):
                ov.append(dict(fate=fate, score=None if sc is None else float(sc), iou=float(it / (G.area + D.area - it)), din=float(it / max(D.area, 1e-9)), gin=float(it / G.area)))
        st['ov'] = ov
        fates = [o['fate'] for o in ov]
        if not ov: st['why'] = 'no_contour' if st['frac'] == 0 else 'contour_tiny_or_elsewhere'
        elif 'kept' in fates:
            k = [o for o in ov if o['fate'] == 'kept']
            best = max(k, key=lambda o: o['iou'])
            if sum(o['din'] > 0.5 for o in k) >= 2: st['why'] = 'kept_split'
            elif best['gin'] > 0.8 and best['din'] < 0.6: st['why'] = 'kept_loose_or_merged'
            elif best['din'] > 0.8: st['why'] = 'kept_tight'
            else: st['why'] = 'kept_shifted'
        elif 'score' in fates: st['why'] = 'dropped_by_box_thresh'
        else: st['why'] = 'dropped_by_min_size'
        recs.append(st)
print(f'{tag}: care GT {len(recs) + len(tp_stats)}, unmatched {len(recs)}')
c = collections.Counter(x['why'] for x in recs)
for k, v in c.most_common():
    xs = [x for x in recs if x['why'] == k]
    print(f"  {k:28s} {v:4d} | median short side(model px) {np.median([x['short'] for x in xs]):5.1f} | median pmax {np.median([x['pmax'] for x in xs]):.3f} | median frac>0.2 {np.median([x['frac'] for x in xs]):.2f} | median lmean {np.median([x['lmean'] for x in xs]):+.2f}")
print('  cross-tab why x taxonomy cat:', dict(collections.Counter((x['why'], x['cat']) for x in recs)))
print(f"  matched GT: median short {np.median([x['short'] for x in tp_stats]):.1f}, median lmean {np.median([x['lmean'] for x in tp_stats]):+.2f}, median frac {np.median([x['frac'] for x in tp_stats]):.2f}")
print('  recall by short side (model px): bin | n | recall')
allg = [(x['short'], 1) for x in tp_stats] + [(x['short'], 0) for x in recs]; a = np.array(allg)
for lo, hi in [(0, 8), (8, 12), (12, 16), (16, 24), (24, 40), (40, 80), (80, 1e9)]:
    s = (a[:, 0] >= lo) & (a[:, 0] < hi)
    if s.sum(): print(f'    [{lo},{hi}) n {int(s.sum()):5d} recall {a[s, 1].mean():.3f}')
sc = [o['score'] for x in recs if x['why'] == 'dropped_by_box_thresh' for o in x['ov'] if o['fate'] == 'score']
if sc: print('  scores of box_thresh-dropped contours on unmatched GT:', np.round(np.sort(sc), 2).tolist()[:60])
json.dump(dict(fn=recs, tp=tp_stats), open(os.path.join(WS, f'fn_diag_{tag}.json'), 'w'))
os.makedirs(os.path.join(WS, 'viz', 'fn_' + tag), exist_ok=True)
r['boxes'] = [postprocess(1 / (1 + np.exp(-r['logits'][i].astype(np.float32))), r['shapes'][i], P) for i in sorted({x['i'] for x in recs})] if False else None
bx = {i: postprocess(1 / (1 + np.exp(-r['logits'][i].astype(np.float32))), r['shapes'][i], P) for i in sorted({x['i'] for x in recs})}
boxes = [bx.get(i, []) for i in range(len(r['names']))]
for k in c:
    items = [(x['i'], r['gts'][x['i']][0][x['g']], f"{x['i']} s{x['short']:.0f} pm{x['pmax']:.2f} f{x['frac']:.2f}") for x in recs if x['why'] == k]
    for s in range(0, min(len(items), 120), 40):
        crop_sheet(r, dd, items[s:s + 40], os.path.join(WS, 'viz', 'fn_' + tag, f'{k}_{s // 40}.jpg'), boxes=boxes)
