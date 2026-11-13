"""H8: is the suppression of missed numbers carried by the deep stage through the neck's top-down path?
Causal test with forward hooks: scale the output of one lateral (neck.ins_conv[k], k=0..3 = in2..in5) by g and
re-run the whole model on the train lab images (all fully-missed-GT images + 600 controls)."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
ix = pickle.load(open(os.path.join(WS, 'fnlab_train_idx.pkl'), 'rb')); keep = sorted(ix['idx'])
lf = os.path.join(WS, 'fnlab_train.txt'); items = read_label_file(os.path.join(VAL_DIR, 'train.txt'))
open(lf, 'w').write(''.join(f'{items[i][0]}\t{items[i][1]}\n' for i in keep))
model = make_model(); G = {'k': -1, 'g': 1.}
for k in range(4): model.neck.ins_conv[k].register_forward_hook(lambda m, i, o, k=k: o * G['g'] if G['k'] == k else None)
base = None
print('lateral x gain | all-images P / R (gt det tp) | boxes on empty images | newly matched GT / lost GT vs base (per image net)')
for k, g in [(-1, 1.)] + [(k, g) for k in range(4) for g in (0., 0.5, 1.5, 2.)]:
    G.update(k=k, g=g); r = run(model, lf); c = r['cnt']; m = prf(c)
    if base is None: base = c
    d = c[:, 2] - base[:, 2]
    print(f"{['none','in2','in3','in4','in5'][k + 1]:4s} x{g:3.1f} | {m['precision']:.4f} / {m['recall']:.4f} ({m['gt']} {m['det']} {m['tp']}) | {int(c[r['empty'], 1].sum())} | +{int(d[d > 0].sum())} / {int(d[d < 0].sum())}", flush=True)
