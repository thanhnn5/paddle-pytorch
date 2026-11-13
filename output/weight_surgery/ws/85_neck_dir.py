"""E4 feasibility: is a thresholded linear detector on the neck output selective for missed text at pixel level?
Features per 1/4-res pixel: fuse (96) and its 3x3 box average (96) -> exactly what one 3x3 conv1 filter with a
centre tap + uniform ring can compute. Fisher direction missed-core (M) vs far background (B), fitted on half of the
train lab images, evaluated on the other half: exceedance of M / detected text (T) / B over thresholds, share of missed
GT with >=30% of the core above threshold, and number of background blobs (>=4 px) above threshold."""
import os, sys, pickle, numpy as np, torch, cv2, pyclipper
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image
from shapely.geometry import Polygon
L0 = float(np.log(0.2 / 0.8))
ix = pickle.load(open(os.path.join(WS, 'fnlab_train_idx.pkl'), 'rb')); idxs = sorted(ix['idx'])
model = make_model(); P = DBPostProcess(**FIXED_PP); F = {}
model.neck.register_forward_hook(lambda m, i, o: F.update({'neck': o}))
model.head.binarize.conv3.register_forward_hook(lambda m, i, o: F.update({'logit': o}))
ds = EvalSet(os.path.join(VAL_DIR, 'train.txt'), VAL_DIR)
def shrink(p, ratio=0.4):
    pg = Polygon(p); d = pg.area * (1 - ratio ** 2) / max(pg.length, 1e-6)
    pc = pyclipper.PyclipperOffset(); pc.AddPath([tuple(x) for x in np.round(p).astype(int)], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return [np.array(s) for s in pc.Execute(-d)]
def feat(f): return torch.cat([f, Fn.avg_pool2d(f, 3, 1, 1, count_include_pad=False)], 1)[0]
D = 192; S = {c: [0, torch.zeros(D, dtype=torch.float64, device='cuda'), torch.zeros(D, D, dtype=torch.float64, device='cuda')] for c in 'MTB'}
store = []
with torch.no_grad():
    for n, i in enumerate(idxs):
        rec = ds[i]; model(rec['image'][None].cuda()); lg = F['logit'][0, 0]; H, W = lg.shape; src_h, src_w = rec['shape'][:2]; h, w = H // 4, W // 4
        boxes = postprocess(torch.sigmoid(lg).cpu().numpy(), rec['shape'], P); dets, gts, _ = analyse_image(boxes, rec['polys'], rec['tags'])
        miss = [g['g'] for g in gts if g['cat'] == 'fn_missed']
        near = np.zeros((h, w), np.uint8); core = np.zeros((h, w), np.int32)
        for g, p in enumerate(rec['polys']):
            pm = np.asarray(p, np.float64) * [w / src_w, h / src_h]; cv2.fillPoly(near, [np.round(pm).astype(np.int32)], 1)
            if g in miss:
                for s in shrink(pm * 4): cv2.fillPoly(core, [np.round(s / 4).astype(np.int32)], g + 1)
        for b in boxes: cv2.fillPoly(near, [np.round(np.asarray(b, np.float64) * [w / src_w, h / src_h]).astype(np.int32)], 1)
        near = cv2.dilate(near, np.ones((5, 5), np.uint8))
        cM = torch.from_numpy(core > 0).cuda(); cT = Fn.max_pool2d((lg > L0).float()[None, None], 4)[0, 0].bool() & ~cM; cB = torch.from_numpy(near == 0).cuda() & ~cT & ~cM
        it = dict(i=i, fuse=F['neck'].half(), core=torch.from_numpy(core).cuda(), cT=cT, cB=cB, fit=n % 2 == 0)
        if it['fit']:
            x = feat(F['neck']).flatten(1).double()
            for c, m in zip('MTB', [cM, cT, cB]):
                xm = x[:, m.flatten()]; S[c][0] += xm.shape[1]; S[c][1] += xm.sum(1); S[c][2] += xm @ xm.T
        store.append(it)
print('fit pixels (1/4 res): M', S['M'][0], 'T', S['T'][0], 'B', S['B'][0])
mu = {c: S[c][1] / S[c][0] for c in 'MTB'}; cov = {c: S[c][2] / S[c][0] - torch.outer(mu[c], mu[c]) for c in 'MTB'}
dirs = {}
for name, Sw, tgt in [('M-vs-B', 0.5 * cov['M'] + 0.5 * cov['B'], mu['M'] - mu['B']), ('M-vs-B, bg cov only', cov['B'], mu['M'] - mu['B']),
                      ('T-vs-B (generic text)', 0.5 * cov['T'] + 0.5 * cov['B'], mu['T'] - mu['B'])]:
    lam = 0.02 * torch.trace(Sw) / D; d = torch.linalg.solve(Sw + lam * torch.eye(D, dtype=torch.float64, device='cuda'), tgt)
    sB = torch.sqrt(d @ cov['B'] @ d); d = d / sB; b = -(d @ mu['B'])   # z-score w.r.t. background
    dirs[name] = (d.cpu().numpy(), float(b))
    df = d.float(); TH = [3, 4, 5, 6, 7, 8, 10, 12]
    for part in ['fit', 'heldout']:
        ex = {c: np.zeros(len(TH)) for c in 'MTB'}; npx = {c: 0 for c in 'MTB'}; gt_hit = np.zeros(len(TH)); ngt = 0; blobs = np.zeros(len(TH)); nimg = 0
        for it in store:
            if it['fit'] != (part == 'fit'): continue
            nimg += 1; z = (feat(it['fuse'].float()).flatten(1).T @ df + float(b)).view(it['core'].shape)
            for c, m in zip('MTB', [it['core'] > 0, it['cT'], it['cB']]):
                v = z[m]; npx[c] += len(v); ex[c] += np.array([float((v > t).sum()) for t in TH])
            for g in torch.unique(it['core'])[1:]:
                v = z[it['core'] == g]; ngt += 1; gt_hit += np.array([float((v > t).float().mean() >= 0.3) for t in TH])
            for k, t in enumerate(TH):
                m = ((z > t) & it['cB']).cpu().numpy().astype(np.uint8)
                if m.any(): n_, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8); blobs[k] += int((st[1:, 4] >= 4).sum())
        print(f'--- {name} [{part}: {nimg} images, {ngt} missed GT, bg px {npx["B"]}]')
        print('    z-thresh   ' + ' '.join(f'{t:8d}' for t in TH))
        print('    M px share ' + ' '.join(f'{v:8.3f}' for v in ex['M'] / max(npx['M'], 1)))
        print('    T px share ' + ' '.join(f'{v:8.3f}' for v in ex['T'] / max(npx['T'], 1)))
        print('    B px share ' + ' '.join(f'{v:8.1e}' for v in ex['B'] / max(npx['B'], 1)))
        print('    missed GT with >=30% core above ' + ' '.join(f'{v:6.3f}' for v in gt_hit / max(ngt, 1)))
        print('    bg blobs (>=4 px) per 100 images ' + ' '.join(f'{v:6.1f}' for v in 100 * blobs / nimg), flush=True)
pickle.dump(dirs, open(os.path.join(WS, 'e4_neck_dirs.pkl'), 'wb'))
