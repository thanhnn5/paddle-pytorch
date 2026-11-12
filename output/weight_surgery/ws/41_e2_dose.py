"""E2: rescale how head.binarize.conv1 reads selected neck channels (conv1.weight[:, c] *= g).
(1) combined effects on the train HeadLab; (2) dose-response on val_dev at the fixed postprocess."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from headlab import *
from bn import eval_sd
DEV = os.path.join(WS, 'val_dev.txt'); sd0 = load_sd()
def edit(gains, sd=None):
    sd = {k: v.clone() for k, v in (sd or sd0).items()}
    for c, g in gains.items(): sd['head.binarize.conv1.weight'][:, c] *= g
    return sd
SUP = [79, 83, 80, 66, 88]; AMP = [38, 77, 68, 50]
cfgs = {'ch79 x0.75': {79: .75}, 'ch79 x0.5': {79: .5}, 'ch79 x0.25': {79: .25}, 'ch79 x0': {79: 0.}, 'ch79 x-0.5': {79: -.5}, 'ch79 x-1': {79: -1.},
        'ch79 x1.5 (reverse)': {79: 1.5}, 'ch79 x2 (reverse)': {79: 2.},
        'ch83 x0': {83: 0.}, 'SUP5 x0': {c: 0. for c in SUP}, 'SUP5 x0.5': {c: .5 for c in SUP},
        'AMP4 x1.5': {c: 1.5 for c in AMP}, 'AMP4 x2': {c: 2. for c in AMP}, 'SUP5 x0 + AMP4 x1.5': {**{c: 0. for c in SUP}, **{c: 1.5 for c in AMP}},
        'ch79 x0 + AMP4 x1.5': {79: 0., **{c: 1.5 for c in AMP}}}
if '--lab' in sys.argv:
    lab = HeadLab(os.path.join(WS, 'train_feat3000.txt'), list(range(0, 300)) + list(range(727, 1127)))
    print('train lab: dTP dFP newbg AUC(mean logit)')
    for nm, g in cfgs.items():
        lab.reset(); w = lab.sd0['conv1.weight'].clone()
        for c, gg in g.items(): w[:, c] *= gg
        lab.head.conv1.weight.data.copy_(w); e = lab.effect(); print(f'{nm:24s} {e[0]:+.3f} {e[1]:+.3f} {e[2]:+7d} {e[3]:.4f}', flush=True)
    del lab; torch.cuda.empty_cache()
base = eval_sd(sd0, DEV, tag='original'); bc = base['cnt']
res = {}
for nm, g in cfgs.items():
    res[nm] = eval_sd(edit(g), DEV, bc, 'E2 ' + nm)['cnt']
pickle.dump(res, open(os.path.join(WS, 'e2_dose_dev.pkl'), 'wb'))
