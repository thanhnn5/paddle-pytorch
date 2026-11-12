"""H1 test: BN recalibration on train images (eval pipeline), dose-response on val_dev."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from bn import *
DEV = os.path.join(WS, 'val_dev.txt'); SUB = os.path.join(WS, 'train_sub800.txt')
sd0 = load_sd()
base = eval_sd(sd0, DEV, tag='original')
bc = base['cnt']
# (a) one-shot measured stats (original upstream), interpolation strength alpha
st = pickle.load(open(os.path.join(WS, 'bn_stats_train800.pkl'), 'rb'))['stats']
for a in [0.25, 0.5, 0.75, 1.0]:
    eval_sd(apply_bn(sd0, st, None, a), DEV, bc, f'one-shot all layers alpha={a}')
# (b) fixed-point recalibration of all executed BN layers
print('fixed-point recalibration, all layers')
sd_fp = recalibrate(sd0, SUB)
save_sd(sd_fp, os.path.join(WS, 'cand_bn_recal_all_train800.pth'))
eval_sd(sd_fp, DEV, bc, 'fixed-point all layers')
# (c) subsets
for nm, f in [('backbone only', lambda n: n.startswith('backbone')), ('s4+neck+head', lambda n: 'blocks_s4' in n or n.startswith('neck') or n.startswith('head')),
              ('neck+head', lambda n: n.startswith('neck') or n.startswith('head')), ('head only', lambda n: n.startswith('head'))]:
    print('fixed-point recalibration,', nm)
    sdx = recalibrate(sd0, SUB, f, rounds=3)
    eval_sd(sdx, DEV, bc, 'fixed-point ' + nm)
for what in ['var', 'mean']:
    print('fixed-point recalibration, all layers, only', what)
    eval_sd(recalibrate(sd0, SUB, None, 3, what), DEV, bc, 'fixed-point all, only ' + what)
