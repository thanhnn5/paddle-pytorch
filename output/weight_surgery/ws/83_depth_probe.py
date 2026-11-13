"""Where is missed text lost? Region-pooled features at each depth (backbone stages, neck levels, head) for
missed GT / matched GT / random background windows on train; LDA with image-grouped 5-fold CV.
AUC(missed vs background) tells whether missed text is still linearly distinguishable from background at that depth."""
import os, sys, pickle, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image
def roc_auc_score(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); o = np.argsort(s, kind='stable'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
class LDA:
    # two-class Fisher discriminant with ridge shrinkage of the pooled within-class covariance (closed form)
    def __init__(self, **kw): pass
    def fit(self, X, y):
        X0, X1 = X[y == 0], X[y == 1]; m0, m1 = X0.mean(0), X1.mean(0)
        S = ((X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1)) / (len(X) - 2); lam = 0.05 * np.trace(S) / len(S)
        self.w = np.linalg.solve(S + lam * np.eye(len(S)), m1 - m0); self.b = -self.w @ (m0 + m1) / 2; return self
    def decision_function(self, X): return X @ self.w + self.b
class GroupKFold:
    def __init__(self, k): self.k = k
    def split(self, X, y, g):
        u = np.unique(g); f = {v: j % self.k for j, v in enumerate(u)}; fo = np.array([f[v] for v in g])
        for j in range(self.k): yield np.flatnonzero(fo != j), np.flatnonzero(fo == j)
rng = np.random.default_rng(0)
ix = pickle.load(open(os.path.join(WS, 'fnlab_train_idx.pkl'), 'rb')); keep = set(ix['idx'])
model = make_model(); P = DBPostProcess(**FIXED_PP); F = {}
model.backbone.register_forward_hook(lambda m, i, o: F.update({f'bb{k}': x for k, x in enumerate(o)}))
model.neck.register_forward_hook(lambda m, i, o: F.update({'neck': o}))
hb = model.head.binarize
hb.conv_bn1.register_forward_hook(lambda m, i, o: F.update({'h1': torch.relu(o)}))
hb.conv_bn2.register_forward_hook(lambda m, i, o: F.update({'h2': torch.relu(o)}))
hb.conv3.register_forward_hook(lambda m, i, o: F.update({'logit': o}))
feats = {}; meta = []
def pool(mask, H, W):
    out = {}
    for k, f in F.items():
        h, w = f.shape[2:]
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
        mt = torch.from_numpy(m).cuda()
        if mt.sum() < 1e-6:
            ys, xs = np.nonzero(mask); mt = torch.zeros(h, w, device='cuda'); mt[min(h - 1, int(ys.mean() * h / H)), min(w - 1, int(xs.mean() * w / W))] = 1
        out[k] = ((f[0] * mt).sum((1, 2)) / mt.sum()).float().cpu().numpy()
    return out
ds = EvalSet(os.path.join(VAL_DIR, 'train.txt'), VAL_DIR)
with torch.no_grad():
    for i in sorted(keep):
        rec = ds[i]; prob = model(rec['image'][None].cuda())['res'][0, 0].cpu().numpy(); H, W = prob.shape; src_h, src_w = rec['shape'][:2]
        boxes = postprocess(prob, rec['shape'], P); dets, gts, _ = analyse_image(boxes, rec['polys'], rec['tags'])
        un = {g['g']: g['cat'] for g in gts}; occ = np.zeros((H, W), np.uint8); regs = []
        for g, (p, t) in enumerate(zip(rec['polys'], rec['tags'])):
            pm = np.round(np.asarray(p, np.float64) * [W / src_w, H / src_h]).astype(np.int32); cv2.fillPoly(occ, [pm], 1)
            if t: continue
            m = np.zeros((H, W), np.float32); cv2.fillPoly(m, [pm], 1.)
            if m.sum() < 20: continue
            c = un.get(g, 'tp')
            if c in ('tp', 'fn_missed'): regs.append((m, 1 if c == 'tp' else 2, cv2.boundingRect(pm)))
        for b in boxes: cv2.fillPoly(occ, [np.round(np.asarray(b, np.float64) * [W / src_w, H / src_h]).astype(np.int32)], 1)
        occ = cv2.dilate(occ, np.ones((15, 15), np.uint8))
        sizes = [r[2][2:] for r in regs] or [(40, 24)]
        for _ in range(40):
            if sum(r[1] == 0 for r in regs) >= 4: break
            bw, bh = sizes[rng.integers(len(sizes))]; bw = min(bw, W - 2); bh = min(bh, H - 2)
            x = rng.integers(0, W - bw); y = rng.integers(0, H - bh)
            if occ[y:y + bh, x:x + bw].any(): continue
            m = np.zeros((H, W), np.float32); m[y:y + bh, x:x + bw] = 1; regs.append((m, 0, None))
        for m, lab, _ in regs:
            pf = pool(m, H, W)
            for k, v in pf.items(): feats.setdefault(k, []).append(v)
            meta.append((i, lab))
meta = np.array(meta); y = meta[:, 1]; grp = meta[:, 0]
print('regions: background', (y == 0).sum(), 'matched GT', (y == 1).sum(), 'missed GT', (y == 2).sum())
print('depth (channels) | AUC missed-vs-bg (CV) | AUC matched-vs-bg (CV) | missed scored by the matched-vs-bg direction: AUC vs bg, share above the bg 99th pct | AUC missed-vs-matched (CV)')
def cv_auc(X, yy, g):
    s = np.zeros(len(yy))
    for tr, te in GroupKFold(5).split(X, yy, g):
        s[te] = LDA(solver='lsqr', shrinkage='auto').fit(X[tr], yy[tr]).decision_function(X[te])
    return roc_auc_score(yy, s), s
for k in ['bb0', 'bb1', 'bb2', 'bb3', 'neck', 'h1', 'h2', 'logit']:
    X = np.stack(feats[k]); X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = X[:, X.std(0) > 0] if X.shape[1] > 1 else X
    a = (y == 0) | (y == 2); b = (y == 0) | (y == 1); c = y > 0
    if X.shape[1] == 1:
        sgn = X[:, 0]; print(f'{k:5s} (1) | {roc_auc_score(y[a] == 2, sgn[a]):.3f} | {roc_auc_score(y[b] == 1, sgn[b]):.3f} | - | {roc_auc_score(y[c] == 2, -sgn[c]):.3f}'); continue
    am, _ = cv_auc(X[a], (y[a] == 2).astype(int), grp[a])
    at, _ = cv_auc(X[b], (y[b] == 1).astype(int), grp[b])
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        trb = tr[y[tr] < 2]; s[te] = LDA(solver='lsqr', shrinkage='auto').fit(X[trb], y[trb]).decision_function(X[te])
    q = np.percentile(s[y == 0], 99)
    ac, _ = cv_auc(X[c], (y[c] == 2).astype(int), grp[c])
    print(f'{k:5s} ({X.shape[1]}) | {am:.3f} | {at:.3f} | {roc_auc_score(y[a] == 2, s[a]):.3f}, {(s[y == 2] > q).mean():.2f} (matched: {(s[y == 1] > q).mean():.2f}) | {ac:.3f}')
pickle.dump(dict(feats={k: np.stack(v) for k, v in feats.items()}, meta=meta), open(os.path.join(WS, 'depth_probe_train.pkl'), 'wb'))
