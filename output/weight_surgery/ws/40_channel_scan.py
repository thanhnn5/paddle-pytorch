"""H3 causal scan: ablate (x0) or double (x2) each of the 96 neck output channels as read by head.binarize.conv1
(conv1.weight[:, c] *= f) and measure the change of the mean logit in TP boxes vs FP boxes on TRAIN images."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from headlab import *
lab = HeadLab(os.path.join(WS, 'train_feat3000.txt'), list(range(0, 300)) + list(range(727, 1127)))
print('baseline AUC(mean logit, TP vs FP) on these train boxes: %.4f' % auc_t(lab.base['m'][lab.base['lab'] >= 0], lab.base['lab'][lab.base['lab'] >= 0]))
W1 = lab.sd0['conv1.weight']; res = {}
for c in range(96):
    for f in (0.0, 2.0):
        lab.reset(); w = W1.clone(); w[:, c] *= f; lab.head.conv1.weight.data.copy_(w)
        res[(c, f)] = lab.effect()
lab.reset()
pickle.dump(res, open(os.path.join(WS, 'channel_scan_train.pkl'), 'wb'))
lev = ['p5', 'p4', 'p3', 'p2']
print('ch level | ablate: dTP dFP sel=dFP-dTP newbg AUC | double: dTP dFP sel newbg AUC   (sorted by ablation selectivity)')
for c in sorted(range(96), key=lambda c: res[(c, 0.0)][1] - res[(c, 0.0)][0]):
    a, d = res[(c, 0.0)], res[(c, 2.0)]
    print(f'{c:2d} {lev[c // 24]} | {a[0]:+.3f} {a[1]:+.3f} {a[1] - a[0]:+.3f} {a[2]:+7d} {a[3]:.4f} | {d[0]:+.3f} {d[1]:+.3f} {d[1] - d[0]:+.3f} {d[2]:+7d} {d[3]:.4f}')
