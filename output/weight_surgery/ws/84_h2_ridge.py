"""E3: closed-form edit of the last layer (head.binarize.conv3, reads the 24 h2 channels) that raises the logit on the
text core of fully missed GT while holding detected text and background still. Fitted on train images only.
Pass 1: second-moment statistics of h2 per pixel class (M missed core, T detected text, B far background).
Solve: ridge regression of target (1 on M, 0 on T and B) -> direction d (24) + offset, scaled so mean d.x on M = 1.
Pass 2: dose-response on the same train images with the real postprocess (optimistic: fit data)."""
import os, sys, pickle, numpy as np, torch, cv2, pyclipper
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image
from shapely.geometry import Polygon
L0 = float(np.log(0.2 / 0.8))
ix = pickle.load(open(os.path.join(WS, 'fnlab_train_idx.pkl'), 'rb')); idxs = sorted(ix['idx']); ctrl = dict(zip(ix['idx'], ix['ctrl']))
model = make_model(); P = DBPostProcess(**FIXED_PP); F = {}; hb = model.head.binarize
hb.conv_bn2.register_forward_hook(lambda m, i, o: F.update({'h2': torch.relu(o)}))
hb.conv3.register_forward_hook(lambda m, i, o: F.update({'logit': o}))
ds = EvalSet(os.path.join(VAL_DIR, 'train.txt'), VAL_DIR)
def shrink(p, ratio=0.4):
    pg = Polygon(p); d = pg.area * (1 - ratio ** 2) / max(pg.length, 1e-6)
    pc = pyclipper.PyclipperOffset(); pc.AddPath([tuple(x) for x in np.round(p).astype(int)], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return [np.array(s) for s in pc.Execute(-d)]
S = {c: [0, torch.zeros(25, dtype=torch.float64, device='cuda'), torch.zeros(25, 25, dtype=torch.float64, device='cuda')] for c in 'MTB'}
store = []
with torch.no_grad():
    for i in idxs:
        rec = ds[i]; out = model(rec['image'][None].cuda()); lg = F['logit'][0, 0]; h2 = F['h2'][0]; H, W = lg.shape; src_h, src_w = rec['shape'][:2]
        pn = torch.sigmoid(lg).cpu().numpy(); boxes = postprocess(pn, rec['shape'], P); dets, gts, _ = analyse_image(boxes, rec['polys'], rec['tags'])
        miss = [g['g'] for g in gts if g['cat'] == 'fn_missed']
        near = np.zeros((H, W), np.uint8); core = np.zeros((H, W), np.uint8)
        for g, p in enumerate(rec['polys']):
            pm = np.asarray(p, np.float64) * [W / src_w, H / src_h]; cv2.fillPoly(near, [np.round(pm).astype(np.int32)], 1)
            if g in miss:
                for s in shrink(pm): cv2.fillPoly(core, [s.astype(np.int32)], 1)
        for b in boxes: cv2.fillPoly(near, [np.round(np.asarray(b, np.float64) * [W / src_w, H / src_h]).astype(np.int32)], 1)
        near = cv2.dilate(near, np.ones((9, 9), np.uint8))
        # classes at h2 resolution (1/2)
        dn = lambda m: torch.from_numpy(cv2.resize(m, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)).cuda().bool()
        cM = dn(core); cT = torch.nn.functional.max_pool2d((lg > L0).float()[None, None], 2)[0, 0].bool() & ~cM; cB = ~dn(near) & ~cT & ~cM
        x = torch.cat([h2.flatten(1).double(), torch.ones(1, h2.shape[1] * h2.shape[2], dtype=torch.float64, device='cuda')], 0)  # 25 x N
        for c, m in zip('MTB', [cM, cT, cB]):
            xm = x[:, m.flatten()]; S[c][0] += xm.shape[1]; S[c][1] += xm.sum(1); S[c][2] += xm @ xm.T
        store.append(dict(i=i, h2=h2.half(), lg=lg.half(), shape=rec['shape'], polys=rec['polys'], tags=rec['tags'], base=score_image(boxes, rec['polys'], rec['tags']),
                          far=torch.from_numpy(near == 0).cuda()))
print('pixels (h2 res): M', S['M'][0], 'T', S['T'][0], 'B', S['B'][0])
out = {}
for name, (wM, wT, wB) in {'bal': (1, 1, 1), 'bgheavy': (1, 1, 4), 'tpfree': (1, 0.05, 4)}.items():
    # class-normalised weighted ridge: each class contributes weight w_c in total
    A = sum(w * S[c][2] / S[c][0] for c, w in zip('MTB', (wM, wT, wB))); b = wM * S['M'][1] / S['M'][0]
    lam = 1e-3 * torch.trace(A[:24, :24]) / 24; R = torch.eye(25, dtype=torch.float64, device='cuda') * lam; R[24, 24] = 0
    d = torch.linalg.solve(A + R, b); mM = float(d @ S['M'][1] / S['M'][0]); d = d / mM
    mu = {c: float(d @ S[c][1] / S[c][0]) for c in 'MTB'}; sd = {c: float(torch.sqrt(d @ (S[c][2] / S[c][0]) @ d - mu[c] ** 2)) for c in 'MTB'}
    w3 = model.head.binarize.conv3.weight[:, 0].flatten(1).mean(1).double()
    print(f"--- {name}: delta per unit alpha: M {mu['M']:+.3f} (sd {sd['M']:.3f}) | T {mu['T']:+.3f} (sd {sd['T']:.3f}) | B {mu['B']:+.4f} (sd {sd['B']:.4f}) | cos(d, conv3 mean tap) {float(torch.nn.functional.cosine_similarity(d[:24], w3, dim=0)):+.3f}")
    out[name] = d.cpu().numpy()
    print('    alpha | lab all-images P / R (gt det tp) | boxes on control-empty images | far-bg px above contour')
    for alpha in [0, 2, 4, 6, 8, 10, 14, 20]:
        cnt = []; farpx = 0; nemp = 0
        for it in store:
            dl = (it['h2'].float().flatten(1).T @ d[:24].float() + float(d[24])).view(it['h2'].shape[1:])
            lg = it['lg'].float() + alpha * torch.nn.functional.interpolate(dl[None, None], scale_factor=2, mode='nearest')[0, 0]
            bx = postprocess(torch.sigmoid(lg).cpu().numpy(), it['shape'], P); c = score_image(bx, it['polys'], it['tags']); cnt.append(c)
            farpx += int(((lg > L0) & it['far']).sum()); nemp += c[1] if len(it['polys']) == 0 else 0
        m = prf(np.array(cnt)); print(f"    {alpha:5.1f} | {m['precision']:.4f} / {m['recall']:.4f} ({m['gt']} {m['det']} {m['tp']}) | {nemp} | {farpx}", flush=True)
pickle.dump(out, open(os.path.join(WS, 'e3_dirs.pkl'), 'wb'))
