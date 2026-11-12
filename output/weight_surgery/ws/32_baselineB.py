"""Baseline B: global knobs on the last layer of the binarize head, evaluated on cached val_dev logits (CPU).
  logit' = s * (logit - L0) + L0 + delta ,  L0 = logit(0.2) = -1.386
  s scales conv3 weight+bias around the thresh-0.2 contour (s only: regions and boxes unchanged, only box scores change);
  delta is a plain bias shift. In weights: conv3.weight *= s ; conv3.bias = s*(bias - L0) + L0 + delta.
For every (s, delta) the full postprocess grid is also swept, to get B alone, and B combined with A."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sweep import *
L0 = float(np.log(0.2 / 0.8))
r = pickle.load(open(os.path.join(WS, 'cache_logits_base_dev.pkl'), 'rb'))
w = evalpy_weights(r['empty'])
TH = [0.1, 0.2, 0.3]; BT = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]; UC = [1.3, 1.4, 1.5]
out = {}
def run_cfg(s, delta):
    rr = dict(r); rr['probs'] = [(1 / (1 + np.exp(-(s * (l.astype(np.float32) - L0) + L0 + delta)))).astype(np.float32) for l in r['logits']]
    res, _ = sweep(rr, TH, BT, UC); return res
print('s, delta | fixed pp (0.2,0.45,1.4) evalpy / all | best over box_thresh at thresh 0.2 unclip 1.4 | best over full grid (evalpy)')
for s, delta in [(1, 0), (0.7, 0), (0.5, 0), (0.35, 0), (0.25, 0), (0.18, 0), (0.12, 0), (1, -1), (1, -2), (1, -3), (1, -4), (0.5, -1), (0.35, -0.5)]:
    res = run_cfg(s, delta); out[(s, delta)] = res
    fx = res[:, 1, 2, 1]
    hb = [(prf(res[:, 1, bi, 1], w)['hmean'], BT[bi]) for bi in range(len(BT))]; bb = max(hb)
    g = max(((prf(res[:, ti, bi, ui], w)['hmean'], TH[ti], BT[bi], UC[ui]) for ti in range(3) for bi in range(len(BT)) for ui in range(3)))
    ga = max(((prf(res[:, ti, bi, ui])['hmean'], TH[ti], BT[bi], UC[ui]) for ti in range(3) for bi in range(len(BT)) for ui in range(3)))
    print(f"s {s:4.2f} d {delta:+.1f} | {prf(fx, w)['hmean']:.4f} / {prf(fx)['hmean']:.4f} (P {prf(fx, w)['precision']:.3f} R {prf(fx, w)['recall']:.3f}) | bt {bb[1]:.2f}: {bb[0]:.4f} | evalpy-best {g[0]:.4f} @ {g[1:]} | all-best {ga[0]:.4f} @ {ga[1:]}", flush=True)
pickle.dump(out, open(os.path.join(WS, 'baselineB_dev.pkl'), 'wb'))
