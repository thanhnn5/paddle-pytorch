"""Baseline A: postprocess sweep on val_dev (cached baseline probability maps)."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sweep import *
TH = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
BT = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
UC = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
res, r = sweep(os.path.join(WS, 'cache_base_val_dev.pkl'), TH, BT, UC)
np.save(os.path.join(WS, 'sweepA_dev.npy'), res)
w = evalpy_weights(r['empty'])
rows = []
for ti, t in enumerate(TH):
    for bi, b in enumerate(BT):
        for ui, u in enumerate(UC):
            c = res[:, ti, bi, ui]
            rows.append((t, b, u, prf(c, w), prf(c)))
print('fixed (0.2,0.45,1.4):', [(fmt(a), fmt(b)) for t, b_, u, a, b in rows if (t, b_, u) == (0.2, 0.45, 1.4)])
for name, k in [('evalpy', 3), ('all', 4)]:
    print(f'--- top 12 by hmean [{name}] (thresh, box_thresh, unclip | evalpy | all)')
    for t, b, u, a, al in sorted(rows, key=lambda x: -x[k]['hmean'])[:12]:
        print(f'{t:.2f} {b:.2f} {u:.1f} | {fmt(a)} | {fmt(al)}')
print('--- marginal: best hmean[evalpy]/[all] for each unclip, each box_thresh, each thresh')
for nm, vals, ix in [('unclip', UC, 2), ('box_thresh', BT, 1), ('thresh', TH, 0)]:
    for v in vals:
        sub = [x for x in rows if x[ix] == v]
        print(nm, v, f"{max(x[3]['hmean'] for x in sub):.4f} {max(x[4]['hmean'] for x in sub):.4f}")
