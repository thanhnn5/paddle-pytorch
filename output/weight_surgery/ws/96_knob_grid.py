"""Two-parameter last-layer family on val_dev at the FIXED postprocess: logit' = s (logit - L0 + c).
s < 1 un-saturates the box score (part 1, edit B); c > 0 lowers the contour to logit > L0 - c (bigger regions, merges).
In weights: conv3.weight *= s ; conv3.bias = s (bias - L0 + c) + L0."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sweep import *
L0 = float(np.log(0.2 / 0.8)); tag = sys.argv[1]
r = pickle.load(open(os.path.join(WS, f'cache_logits_base_{tag}.pkl'), 'rb')); w = evalpy_weights(r['empty'])
def counts(s, c):
    rr = dict(r); rr['probs'] = [(1 / (1 + np.exp(-(s * (l.astype(np.float32) - L0 + c) + L0)))).astype(np.float32) for l in r['logits']]
    return sweep(rr, [0.2], [0.45], [1.4])[0][:, 0, 0, 0]
base = counts(1, 0); mb = prf(base); out = {}
S = [1, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2]; C = [0, 1, 2, 3, 4, 5, 6, 8]
print(f'{tag}: all-images P / R, GT gained-lost; baseline {mb["precision"]:.4f} / {mb["recall"]:.4f} det {mb["det"]}')
print('s \\ c   ' + ''.join(f'{c:>22}' for c in C))
for s in S:
    line = f'{s:5.2f}   '
    for c in C:
        k = counts(s, c); out[(s, c)] = k; m = prf(k); d = k[:, 2] - base[:, 2]
        line += f"  {m['precision']:.3f}/{m['recall']:.4f} +{int(d[d > 0].sum())}-{int(-d[d < 0].sum())}".ljust(22)
    print(line, flush=True)
pickle.dump(out, open(os.path.join(WS, f'knob_grid_{tag}.pkl'), 'wb'))
