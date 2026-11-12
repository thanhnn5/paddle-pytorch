"""Head-only laboratory: cache neck outputs (fuse) for a set of images on the GPU, with baseline boxes labelled
TP / FP, then evaluate any edit of the binarize head by re-running only the head. Reports the change of the
mean logit inside baseline boxes (TP vs FP) and the number of background pixels pushed above the thresh contour."""
import os, sys, pickle, numpy as np, torch, cv2
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image
L0 = float(np.log(0.2 / 0.8))


class HeadLab:
    def __init__(self, label_file, idxs, sd=None):
        self.model = make_model(sd)
        self.head = self.model.head.binarize
        self.sd0 = {k: v.clone() for k, v in self.head.state_dict().items()}
        ds = EvalSet(label_file, VAL_DIR); P = DBPostProcess(**dict(FIXED_PP, box_thresh=0.3))
        self.items = []; fz = {}
        hk = self.model.neck.register_forward_hook(lambda m, i, o: fz.__setitem__('f', o))
        lg = {}
        hk2 = self.head.conv3.register_forward_hook(lambda m, i, o: lg.__setitem__('l', o))
        with torch.no_grad():
            for i in idxs:
                rec = ds[i]
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
                dets, gts, _ = analyse_image(boxes, rec['polys'], rec['tags']) if boxes else ([], [], [])
                cat = {d['d']: d['cat'] for d in dets}
                idmap = np.zeros((H, W), np.int32); near = np.zeros((H, W), np.uint8); labs = [-1]
                for d, pts in enumerate(minis):
                    cv2.fillPoly(idmap, pts.reshape(1, -1, 2).astype('int32'), d + 1)
                    cv2.fillPoly(near, [P.unclip(pts, 2.0).reshape(-1, 2).astype(np.int32)], 1)
                    c = cat.get(d, 'tp'); labs.append(1 if c == 'tp' else 0 if c in ('fp_background', 'fp_empty_image') else -1)
                for p in rec['polys']:
                    cv2.fillPoly(near, [(np.asarray(p, np.float64) * [W / src_w, H / src_h]).astype(np.int32)], 1)
                self.items.append(dict(fuse=fz['f'].half(), idmap=torch.from_numpy(idmap).cuda().flatten().long(), labs=torch.tensor(labs).cuda(),
                                       far=torch.from_numpy(near == 0).cuda().flatten(), base=lg['l'][0, 0].flatten().clone()))
        hk.remove(); hk2.remove()
        nb = torch.cat([it['labs'][1:] for it in self.items])
        print(f'HeadLab: {len(self.items)} images, boxes TP {(nb == 1).sum().item()} FP {(nb == 0).sum().item()} other {(nb == -1).sum().item()}')
        self.base = self.measure()

    def reset(self):
        self.head.load_state_dict(self.sd0)

    @torch.no_grad()
    def box_logits(self):
        outs, lab, newbg = [], [], 0
        for it in self.items:
            l = self.head.conv3(F.relu(self.head.conv_bn2(self.head.conv2(F.relu(self.head.conv_bn1(self.head.conv1(it['fuse'].float())))))))[0, 0].flatten()
            K = len(it['labs'])
            s = torch.zeros(K, device='cuda').index_add_(0, it['idmap'], l); n = torch.zeros(K, device='cuda').index_add_(0, it['idmap'], torch.ones_like(l))
            outs.append((s / n.clamp_min(1))[1:]); lab.append(it['labs'][1:])
            newbg += int(((l > L0) & it['far']).sum())
        return torch.cat(outs), torch.cat(lab), newbg

    def measure(self):
        m, lab, newbg = self.box_logits()
        return dict(m=m, lab=lab, newbg=newbg)

    def effect(self):
        """(mean change of box mean-logit on TP boxes, on FP boxes, change in far-background pixels above contour, AUC of mean logit)"""
        r = self.measure(); d = r['m'] - self.base['m']; lab = r['lab']
        return float(d[lab == 1].mean()), float(d[lab == 0].mean()), r['newbg'] - self.base['newbg'], auc_t(r['m'][lab >= 0], lab[lab >= 0])


def auc_t(s, y):
    s = s.double().cpu().numpy(); y = y.cpu().numpy()
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
