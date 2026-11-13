"""Recall baseline A': precision/recall frontier of the postprocess alone on cached logits.
usage: 81_frontier.py <logits_cache.pkl> <tag>"""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sweep import *
cache, tag = sys.argv[1:3]
r = pickle.load(open(cache, 'rb')); w = evalpy_weights(r['empty'])
r['probs'] = [(1 / (1 + np.exp(-l.astype(np.float32)))).astype(np.float32) for l in r['logits']]
TH = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]; BT = [0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]; UC = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]
res, _ = sweep(r, TH, BT, UC, procs=6)
np.save(os.path.join(WS, f'frontier_{tag}.npy'), res)
rows = [(t, b, u, prf(res[:, ti, bi, ui], w), prf(res[:, ti, bi, ui])) for ti, t in enumerate(TH) for bi, b in enumerate(BT) for ui, u in enumerate(UC)]
fx = [x for x in rows if x[:3] == (0.2, 0.45, 1.4)][0]
print('fixed:', fmt(fx[3]), '| all:', fmt(fx[4]))
print('--- unclip 1.4 only: thresh x box_thresh, evalpy P/R (det count)')
for t in TH:
    print(f'th {t:.2f}: ' + '  '.join(f"bt{b:.2f} {a['precision']:.3f}/{a['recall']:.4f}({a['det']})" for tt, b, u, a, al in rows if tt == t and u == 1.4 and b <= 0.5))
print('--- thresh 0.2, box_thresh 0.45: unclip, evalpy P/R/H | all P/R')
for tt, b, u, a, al in rows:
    if tt == 0.2 and b == 0.45: print(f"uc {u:.1f}: {a['precision']:.4f}/{a['recall']:.4f}/{a['hmean']:.4f} | {al['precision']:.4f}/{al['recall']:.4f}")
for nm, k, ucs in [('evalpy, unclip 1.4', 3, [1.4]), ('evalpy, any unclip', 3, UC), ('all, unclip 1.4', 4, [1.4])]:
    print(f'--- Pareto frontier [{nm}]: recall, precision, hmean @ (thresh, box_thresh, unclip)')
    pts = sorted([x for x in rows if x[2] in ucs], key=lambda x: (-x[k]['recall'], -x[k]['precision'])); best = -1
    for x in pts:
        if x[k]['precision'] > best + 1e-9:
            best = x[k]['precision']; print(f"  R {x[k]['recall']:.4f} P {x[k]['precision']:.4f} H {x[k]['hmean']:.4f} det {x[k]['det']} @ {x[:3]}")
