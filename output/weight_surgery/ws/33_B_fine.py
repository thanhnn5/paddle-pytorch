"""Baseline B, finer scale grid on val_dev at thresh 0.2 / unclip 1.4 (boxes identical to the original), box_thresh varied."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sweep import *
L0 = float(np.log(0.2 / 0.8))
r = pickle.load(open(os.path.join(WS, 'cache_logits_base_dev.pkl'), 'rb')); w = evalpy_weights(r['empty'])
BT = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
base = None
for s in [1.0, 0.4, 0.3, 0.27, 0.25, 0.23, 0.2]:
    rr = dict(r); rr['probs'] = [(1 / (1 + np.exp(-(s * (l.astype(np.float32) - L0) + L0)))).astype(np.float32) for l in r['logits']]
    res, _ = sweep(rr, [0.2], BT, [1.4]); res = res[:, 0, :, 0]
    if base is None: base = res[:, 3]
    print(f's {s:.2f} | ' + ' '.join(f"{b:.2f}:{prf(res[:, i], w)['hmean']:.4f}/{prf(res[:, i])['hmean']:.4f}" for i, b in enumerate(BT)), flush=True)
    c = res[:, 3]
    print(f"        fixed pp: evalpy {fmt(prf(c, w))} | {fmt_boot(paired_bootstrap(base, c, w))}\n        fixed pp: all    {fmt(prf(c))} | {fmt_boot(paired_bootstrap(base, c))}", flush=True)
