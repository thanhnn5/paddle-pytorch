"""Causal scan for recall: for each of the 96 neck channels read by head.binarize.conv1, set gain 0 / 2 / 3 and measure
the logit change inside fully missed GT (train), in TP boxes, FP boxes, and far-background pixels above the contour."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from fnlab import *
rng = np.random.default_rng(0)
items = read_label_file(os.path.join(VAL_DIR, 'train.txt'))
empty = np.array([len(json.loads(l)) == 0 for _, l in items])
ctrl = list(rng.choice(np.flatnonzero(empty), 250, replace=False)) + list(rng.choice(np.flatnonzero(~empty), 350, replace=False))
lab = FNLab(os.path.join(VAL_DIR, 'train.txt'), ctrl)
pickle.dump(dict(idx=[it['i'] for it in lab.items], ctrl=[it['ctrl'] for it in lab.items]), open(os.path.join(WS, 'fnlab_train_idx.pkl'), 'wb'))
W0 = lab.sd0['conv1.weight']; res = {}
lev = lambda c: ['p5', 'p4', 'p3', 'p2'][c // 24]
for c in range(96):
    for g in [0., 2., 3.]:
        w = W0.clone(); w[:, c] *= g; lab.head.conv1.weight.data.copy_(w); res[(c, g)] = lab.effect()
lab.reset()
pickle.dump(res, open(os.path.join(WS, 'fn_scan_train.pkl'), 'wb'))
def show(title, key, g, n=14):
    print(title); print('  ch(level) | dMiss  rec  | dTP   dFP   | far-bg px')
    for c in sorted(range(96), key=lambda c: -res[(c, g)]['dmiss'])[:n]:
        e = res[(c, g)]; print(f"  {c:2d} ({lev(c)}) | {e['dmiss']:+.2f} {e['rec']:.3f} | {e['dtp']:+.2f} {e['dfp']:+.2f} | {e['dbg']:+d}")
show('--- ablate (gain 0): channels whose removal RAISES missed-GT logits most (suppressors)', 'dmiss', 0.)
show('--- double (gain 2): channels whose amplification raises missed-GT logits most', 'dmiss', 2.)
show('--- gain 3', 'dmiss', 3.)
print('--- level gains: all 24 channels of one level x g')
for L in range(4):
    for g in [0., 0.5, 1.5, 2., 3.]:
        w = W0.clone(); w[:, L * 24:(L + 1) * 24] *= g; lab.head.conv1.weight.data.copy_(w); e = lab.effect()
        print(f"  {['p5','p4','p3','p2'][L]} x{g}: dMiss {e['dmiss']:+.2f} rec {e['rec']:.3f} dTP {e['dtp']:+.2f} dFP {e['dfp']:+.2f} bg {e['dbg']:+d}")
lab.reset()
