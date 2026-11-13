"""H9: are blanks an input-statistics problem (dark / low-contrast)? Apply x' = a*x + b to the normalised input
(foldable into the first conv + its BN) and measure on val_dev."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lf = os.path.join(WS, 'val_dev.txt'); model = make_model(); G = dict(a=1., b=0.)
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(x * G['a'] + G['b'])
wm = Wrap(model); base = None
print('a b | evalpy P / R | all P / R (det) | gained / lost')
for a, b in [(1, 0), (1.25, 0), (1.5, 0), (0.8, 0), (1, 0.3), (1, -0.3), (1.25, 0.3), (2, 0)]:
    G.update(a=a, b=b); r = run(wm, lf); c = r['cnt']; w = evalpy_weights(r['empty']); e = prf(c, w); m = prf(c)
    if base is None: base = c
    d = c[:, 2] - base[:, 2]
    print(f"{a} {b} | {e['precision']:.4f} / {e['recall']:.4f} | {m['precision']:.4f} / {m['recall']:.4f} ({m['det']}) | +{int(d[d > 0].sum())} / {int(d[d < 0].sum())}", flush=True)
