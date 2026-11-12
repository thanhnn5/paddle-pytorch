"""Error taxonomy from cached predictions. Categories for unmatched detections (FP) and unmatched GT (FN)."""
import os, sys, pickle, json, collections, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from shapely.geometry import Polygon


def poly(p):
    q = Polygon(np.asarray(p, float))
    return q if q.is_valid else q.buffer(0)


def analyse_image(boxes, polys, tags):
    """Replicates DetectionIoUEvaluator matching; returns per-det and per-gt records."""
    G = [poly(p) for p in polys]; D = [poly(b) for b in boxes]
    ng, nd = len(G), len(D)
    inter = np.zeros((ng, nd)); iou = np.zeros((ng, nd))
    for g in range(ng):
        for d in range(nd):
            it = G[g].intersection(D[d]).area
            inter[g, d] = it
            un = G[g].area + D[d].area - it
            iou[g, d] = it / un if un > 0 else 0
    ga = np.array([g.area for g in G]); da = np.array([d.area for d in D])
    gdc = [g for g in range(ng) if tags[g]]
    ddc = [d for d in range(nd) if any(da[d] > 0 and inter[g, d] / da[d] > 0.5 for g in gdc)]
    gm = -np.ones(ng, int); dm = -np.ones(nd, int)
    for g in range(ng):
        for d in range(nd):
            if gm[g] < 0 and dm[d] < 0 and g not in gdc and d not in ddc and iou[g, d] > 0.5:
                gm[g] = d; dm[d] = g
    care_g = [g for g in range(ng) if g not in gdc]
    din = inter / np.maximum(da[None], 1e-9)   # fraction of det inside gt
    gin = inter / np.maximum(ga[:, None], 1e-9)  # fraction of gt covered by det
    dets = []
    for d in range(nd):
        if d in ddc or dm[d] >= 0:
            continue
        rec = dict(d=d, area=float(da[d]))
        if len(care_g) == 0 and ng == 0:
            rec['cat'] = 'fp_empty_image'
        else:
            ov = [g for g in care_g if iou[g, d] > 0.05 or din[g, d] > 0.3]
            best = max(care_g, key=lambda g: iou[g, d]) if care_g else -1
            rec['best_iou'] = float(iou[best, d]) if best >= 0 else 0.
            rec['g'] = int(best)
            covered = [g for g in care_g if gin[g, d] > 0.5]
            if not ov:
                rec['cat'] = 'fp_background'
            elif len(covered) >= 2:
                rec['cat'] = 'merge'
            elif iou[best, d] > 0.5:
                rec['cat'] = 'duplicate'
            else:
                sib = [e for e in range(nd) if e != d and e not in ddc and din[best, e] > 0.5]
                if din[best, d] > 0.6 and len(sib) >= 1:
                    rec['cat'] = 'split'   # one of several dets lying inside the same GT
                elif iou[best, d] >= 0.3:
                    if gin[best, d] > 0.8:
                        rec['cat'] = 'loose'
                    elif din[best, d] > 0.8:
                        rec['cat'] = 'tight'
                    else:
                        rec['cat'] = 'shifted'
                elif din[best, d] > 0.6:
                    rec['cat'] = 'partial_inside_gt'  # small fragment inside a GT, IoU<0.3
                elif gin[best, d] > 0.8:
                    rec['cat'] = 'very_loose'
                else:
                    rec['cat'] = 'low_overlap'
        dets.append(rec)
    gts = []
    for g in care_g:
        if gm[g] >= 0:
            continue
        rec = dict(g=g, area=float(ga[g]))
        if nd == 0 or iou[g].max() <= 0.05 and gin[g].max() <= 0.3:
            rec['cat'] = 'fn_missed'
        else:
            d = int(np.argmax(iou[g])); rec['best_iou'] = float(iou[g, d]); rec['d'] = d
            inside = [e for e in range(nd) if din[g, e] > 0.5]
            dmerge = [e for e in range(nd) if gin[g, e] > 0.5 and sum(gin[h, e] > 0.5 for h in care_g) >= 2]
            if dmerge:
                rec['cat'] = 'fn_merged'
            elif len(inside) >= 2:
                rec['cat'] = 'fn_split'
            elif iou[g, d] > 0.5:
                rec['cat'] = 'fn_greedy'
            elif iou[g, d] >= 0.3:
                rec['cat'] = 'fn_geometry_0.3-0.5'
            else:
                rec['cat'] = 'fn_partial_<0.3'
        gts.append(rec)
    tp_iou = [float(iou[g, gm[g]]) for g in care_g if gm[g] >= 0]
    return dets, gts, tp_iou


def run_taxonomy(r, boxes=None):
    boxes = boxes if boxes is not None else r['boxes']
    out = []
    for i in range(len(boxes)):
        polys, tags = r['gts'][i]
        d, g, t = analyse_image(boxes[i], polys, tags)
        out.append(dict(i=i, dets=d, gts=g, tp_iou=t))
    return out


def summarize(tax):
    cd = collections.Counter(x['cat'] for t in tax for x in t['dets'])
    cg = collections.Counter(x['cat'] for t in tax for x in t['gts'])
    return cd, cg


def crop_sheet(r, data_dir, items, path, boxes=None, cell=(360, 240), cols=5, pad=0.6, maxn=40):
    """items: list of (image_index, focus_polygon, title). Draw GT green, ignore yellow, preds red."""
    boxes = boxes if boxes is not None else r['boxes']
    cells = []
    for i, focus, title in items[:maxn]:
        img = cv2.imread(os.path.join(data_dir, r['names'][i]))
        polys, tags = r['gts'][i]
        vis = img.copy()
        lw = max(1, int(round(max(img.shape[:2]) / 500)))
        for p, t in zip(polys, tags):
            cv2.polylines(vis, [np.asarray(p, np.int32)], True, (0, 255, 255) if t else (0, 220, 0), lw)
        for b in boxes[i]:
            cv2.polylines(vis, [np.asarray(b, np.int32)], True, (0, 0, 255), lw)
        f = np.asarray(focus, float)
        x0, y0 = f.min(0); x1, y1 = f.max(0)
        w, h = x1 - x0, y1 - y0
        m = max(pad * max(w, h), 20)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w = max(w / 2 + m, (h / 2 + m) * cell[0] / cell[1]); half_h = half_w * cell[1] / cell[0]
        X0, X1 = int(max(0, cx - half_w)), int(min(img.shape[1], cx + half_w))
        Y0, Y1 = int(max(0, cy - half_h)), int(min(img.shape[0], cy + half_h))
        c = vis[Y0:Y1, X0:X1]
        s = min(cell[0] / c.shape[1], cell[1] / c.shape[0])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
        canvas = np.full((cell[1] + 16, cell[0], 3), 40, np.uint8)
        canvas[16:16 + c.shape[0], :c.shape[1]] = c
        cv2.putText(canvas, title[:52], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cells.append(canvas)
    if not cells:
        return
    rows = []
    for k in range(0, len(cells), cols):
        row = cells[k:k + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(path, np.vstack(rows))


if __name__ == '__main__':
    cache = sys.argv[1]; tag = sys.argv[2]
    r = pickle.load(open(cache, 'rb')); r['probs'] = None
    tax = run_taxonomy(r)
    cd, cg = summarize(tax)
    N = len(tax); m = prf(r['cnt'])
    print('images', N, 'empty', int(r['empty'].sum()), fmt(m))
    print('unmatched dets:', sum(cd.values())); [print(f'  {k:22s} {v}') for k, v in cd.most_common()]
    print('unmatched gts :', sum(cg.values())); [print(f'  {k:22s} {v}') for k, v in cg.most_common()]
    ti = np.array([x for t in tax for x in t['tp_iou']])
    print('TP IoU: mean %.3f, share in (0.5,0.6] %.3f, (0.6,0.7] %.3f' % (ti.mean(), (ti <= 0.6).mean(), ((ti > 0.6) & (ti <= 0.7)).mean()))
    json.dump(tax, open(os.path.join(WS, f'taxonomy_{tag}.json'), 'w'))
    os.makedirs(os.path.join(WS, 'viz', tag), exist_ok=True)
    rng = np.random.default_rng(0)
    for cat in list(cd) + list(cg):
        items = []
        for t in tax:
            i = t['i']
            for x in t['dets']:
                if x['cat'] == cat:
                    items.append((i, r['boxes'][i][x['d']], f"{i} {cat} iou{ x.get('best_iou', 0):.2f}"))
            for x in t['gts']:
                if x['cat'] == cat:
                    items.append((i, r['gts'][i][0][x['g']], f"{i} {cat} iou{x.get('best_iou', 0):.2f}"))
        sel = rng.permutation(len(items))[:30]
        crop_sheet(r, VAL_DIR, [items[k] for k in sel], os.path.join(WS, 'viz', tag, f'{cat}.jpg'))
