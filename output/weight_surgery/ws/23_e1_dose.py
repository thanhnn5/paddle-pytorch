"""E1: add the train-fitted TP-vs-FP LDA direction (in h2 space) to the final layer of the binarize head.
  conv3.weight[c, 0, a, b] += alpha * w_c  (all four taps),  conv3.bias += alpha * b0
Two bias conventions: 'tp0' (b0 = -w.m_TP: mean logit of train TP boxes unchanged; pure re-ranking, FP go down)
and 'mid' (LDA midpoint: TP up, FP down). Dose-response over alpha on val_dev, fixed postprocess."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from bn import eval_sd
from common import *
exec(open(os.path.join(os.path.dirname(__file__), '21_linear_sep.py')).read().split("print('train boxes'")[0])
DEV = os.path.join(WS, 'val_dev.txt')
Xt = tr['X']['h2'][kt]; w, bmid = lda(Xt, yt, 0.1); m1 = Xt[yt == 1].mean(0)
pickle.dump(dict(w=w, bmid=bmid, btp0=-(w @ m1)), open(os.path.join(WS, 'e1_dir_h2.pkl'), 'wb'))
sd0 = load_sd()
def edit(alpha, b0):
    sd = {k: v.clone() for k, v in sd0.items()}
    sd['head.binarize.conv3.weight'] += alpha * torch.tensor(w, dtype=torch.float32)[:, None, None, None]
    sd['head.binarize.conv3.bias'] += alpha * float(b0)
    return sd
base = eval_sd(sd0, DEV, tag='original'); bc = base['cnt']
for conv, b0 in [('tp0', -(w @ m1)), ('mid', bmid)]:
    for a in [0.1, 0.25, 0.5, 1.0, 2.0]:
        sd = edit(a, b0)
        eval_sd(sd, DEV, bc, f'E1 {conv} alpha={a}')
        save_sd(sd, os.path.join(WS, f'cand_e1_h2lda_{conv}_a{a}.pth'), sd0)
