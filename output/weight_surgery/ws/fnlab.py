"""Recall lab: cache neck outputs for train images that contain a fully missed GT (no detection overlapping it)
plus control images; evaluate any edit of the binarize head by re-running only the head.
Measures: logit inside missed GT (mean of the top 20% pixels of each polygon), share of missed GT whose top-20% mean
crosses the thresh contour, mean logit in baseline TP / FP boxes, far-background pixels above the contour."""
import os, sys, pickle, numpy as np, torch, cv2
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image
L0 = float(np.log(0.2 / 0.8))


class FNLab:
    def __init__(self, label_file, control_idx, data_dir=VAL_DIR, sd=None, limit=None, keep_all_miss=True, keep_bb=False):
        self.model = make_model(sd); self.head = self.model.head.binarize
        self.sd0 = {k: v.clone() for k, v in self.head.state_dict().items()}
        P = DBPostProcess(**FIXED_PP); control_idx = set(control_idx)
        self.items = []; fz = {}; lg = {}
        hk = self.model.neck.register_forward_hook(lambda m, i, o: fz.__setitem__('f', o))
        hk3 = self.model.backbone.register_forward_hook(lambda m, i, o: fz.__setitem__('bb', [x.half() for x in o])) if keep_bb else None
        hk2 = self.head.conv3.register_forward_hook(lambda m, i, o: lg.__setitem__('l', o))
        dl = loader(label_file, data_dir); n = 0
        with torch.no_grad():
            for rec in dl:
                i = rec['idx']
                if limit and i >= limit: break
                prob = self.model(rec['image'][None].cuda())['res'][0, 0]; pn = prob.cpu().numpy(); H, W = pn.shape; src_h, src_w = rec['shape'][:2]
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
                dets, gts, _ = analyse_image(boxes, rec['polys'], rec['tags'])
                miss = [g['g'] for g in gts if g['cat'] == 'fn_missed']
                if not ((miss and keep_all_miss) or i in control_idx): continue
                cat = {d['d']: d['cat'] for d in dets}
                idmap = np.zeros((H, W), np.int32); near = np.zeros((H, W), np.uint8); labs = [-1]
                for d, pts in enumerate(minis):
                    cv2.fillPoly(idmap, pts.reshape(1, -1, 2).astype('int32'), d + 1)
                    cv2.fillPoly(near, [P.unclip(pts, 2.0).reshape(-1, 2).astype(np.int32)], 1)
                    c = cat.get(d, 'tp'); labs.append(1 if c == 'tp' else 0 if c in ('fp_background', 'fp_empty_image') else -1)
                for p in rec['polys']:
                    cv2.fillPoly(near, [(np.asarray(p, np.float64) * [W / src_w, H / src_h]).astype(np.int32)], 1)
                mm = []
                for g in miss:
                    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(np.asarray(rec['polys'][g], np.float64) * [W / src_w, H / src_h]).astype(np.int32)], 1)
                    ix = np.flatnonzero(m.ravel())
                    if len(ix) >= 20: mm.append(torch.from_numpy(ix).cuda())
                self.items.append(dict(i=i, fuse=fz['f'].half(), idmap=torch.from_numpy(idmap).cuda().flatten().long(), labs=torch.tensor(labs).cuda(),
                                       far=torch.from_numpy(near == 0).cuda().flatten(), miss=mm, ctrl=i in control_idx, bb=fz.get('bb')))
        hk.remove(); hk2.remove()
        nb = torch.cat([it['labs'][1:] for it in self.items])
        print(f'FNLab: {len(self.items)} images, missed GT {sum(len(it["miss"]) for it in self.items)}, boxes TP {(nb == 1).sum().item()} FP {(nb == 0).sum().item()}')
        self.base = self.measure()
        print('  base: missed top20 logit mean %.2f, TP box %.2f, FP box %.2f, far px above contour %d' % (self.base['miss'].mean(), self.base['m'][self.base['lab'] == 1].mean(), self.base['m'][self.base['lab'] == 0].mean(), self.base['newbg']))

    def reset(self):
        self.head.load_state_dict(self.sd0)

    def logits(self, it):
        h = self.head
        return h.conv3(F.relu(h.conv_bn2(h.conv2(F.relu(h.conv_bn1(h.conv1(it['fuse'].float())))))))[0, 0].flatten()

    @torch.no_grad()
    def measure(self):
        outs, lab, miss, newbg = [], [], [], 0
        for it in self.items:
            l = self.logits(it); K = len(it['labs'])
            s = torch.zeros(K, device='cuda').index_add_(0, it['idmap'], l); n = torch.zeros(K, device='cuda').index_add_(0, it['idmap'], torch.ones_like(l))
            outs.append((s / n.clamp_min(1))[1:]); lab.append(it['labs'][1:])
            newbg += int(((l > L0) & it['far']).sum())
            for ix in it['miss']:
                v = l[ix]; miss.append(v.topk(max(1, len(ix) // 5)).values.mean())
        return dict(m=torch.cat(outs), lab=torch.cat(lab), miss=torch.stack(miss), newbg=newbg)

    def effect(self):
        r = self.measure(); d = r['m'] - self.base['m']; lab = r['lab']
        return dict(dmiss=float((r['miss'] - self.base['miss']).mean()), rec=float((r['miss'] > L0).float().mean()),
                    dtp=float(d[lab == 1].mean()), dfp=float(d[lab == 0].mean()), dbg=r['newbg'] - self.base['newbg'])
