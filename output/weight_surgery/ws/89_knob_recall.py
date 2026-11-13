"""Recall baseline B': bias shift / scale of the last layer at the FIXED postprocess, on cached val_dev logits.
logit' = s * (logit - L0) + L0 + delta."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sweep import *
L0 = float(np.log(0.2 / 0.8))
r = pickle.load(open(os.path.join(WS, 'cache_logits_base_dev.pkl'), 'rb')); w = evalpy_weights(r['empty']); base = None
print('s delta | evalpy P / R / H | all P / R (det) | dR, dP all [95% CI]')
for s, delta in [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (2, 0), (4, 0), (2, 3), (0.5, 3), (0.5, 4)]:
    rr = dict(r); rr['probs'] = [(1 / (1 + np.exp(-(s * (l.astype(np.float32) - L0) + L0 + delta)))).astype(np.float32) for l in r['logits']]
    res, _ = sweep(rr, [0.2], [0.45], [1.4]); c = res[:, 0, 0, 0]
    if base is None: base = c
    a = prf(c, w); m = prf(c); bt = paired_bootstrap(base, c, B=2000)
    print(f"{s:3.1f} {delta:+.0f} | {a['precision']:.4f} / {a['recall']:.4f} / {a['hmean']:.4f} | {m['precision']:.4f} / {m['recall']:.4f} ({m['det']}) | dR {bt['recall'][0]:+.4f} [{bt['recall'][1]:+.4f},{bt['recall'][2]:+.4f}] dP {bt['precision'][0]:+.4f} [{bt['precision'][1]:+.4f},{bt['precision'][2]:+.4f}]", flush=True)
