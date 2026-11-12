"""Does E2 (neck-channel rescaling in head.binarize.conv1) add on top of baseline B (conv3 scale s about L0)?
val_dev, fixed postprocess; paired bootstrap against B alone."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from bn import eval_sd
from common import *
L0 = float(np.log(0.2 / 0.8)); DEV = os.path.join(WS, 'val_dev.txt'); sd0 = load_sd()
def editB(sd, s, delta=0.0):
    sd = {k: v.clone() for k, v in sd.items()}
    sd['head.binarize.conv3.weight'] *= s
    sd['head.binarize.conv3.bias'] = s * (sd['head.binarize.conv3.bias'] - L0) + L0 + delta
    return sd
def editG(sd, gains):
    sd = {k: v.clone() for k, v in sd.items()}
    for c, g in gains.items(): sd['head.binarize.conv1.weight'][:, c] *= g
    return sd
SUP = [79, 83, 80, 66, 88]; AMP = [38, 77, 68, 50]
S = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
base = eval_sd(sd0, DEV, tag='original')['cnt']
B = eval_sd(editB(sd0, S), DEV, base, f'B s={S}')['cnt']
print('--- below: paired differences are against B alone')
res = {'B': B}
for nm, g in {'ch79 x0.5': {79: .5}, 'ch79 x0': {79: 0.}, 'ch79 x-0.5': {79: -.5}, 'ch79 x-1': {79: -1.}, 'ch79 x-2': {79: -2.}, 'ch79 x2 (reverse)': {79: 2.},
              'SUP5 x0': {c: 0. for c in SUP}, 'SUP5 x-0.5': {c: -.5 for c in SUP}, 'AMP4 x2': {c: 2. for c in AMP},
              'SUP5 x0 + AMP4 x1.5': {**{c: 0. for c in SUP}, **{c: 1.5 for c in AMP}}, 'SUP5 x0 + AMP4 x2': {**{c: 0. for c in SUP}, **{c: 2. for c in AMP}}}.items():
    res[nm] = eval_sd(editB(editG(sd0, g), S), DEV, B, f'B s={S} + {nm}')['cnt']
pickle.dump(res, open(os.path.join(WS, f'e2_on_B{S}_dev.pkl'), 'wb'))
