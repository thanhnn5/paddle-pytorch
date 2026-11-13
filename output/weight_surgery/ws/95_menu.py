"""Recall menu on dev / holdout / full val from cached baseline logits: last-layer knob at the FIXED postprocess
(logit' = s (logit - L0) + L0 + delta) and postprocess-only settings. Paired bootstrap vs the original at fixed pp."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sweep import *
L0 = float(np.log(0.2 / 0.8))
R = {k: pickle.load(open(os.path.join(WS, f'cache_logits_base_{k}.pkl'), 'rb')) for k in ['dev', 'holdout']}
def counts(r, s, delta, th, bt, uc=1.4):
    rr = dict(r); rr['probs'] = [(1 / (1 + np.exp(-(s * (l.astype(np.float32) - L0) + L0 + delta)))).astype(np.float32) for l in r['logits']]
    return sweep(rr, [th], [bt], [uc])[0][:, 0, 0, 0]
rows = [('original', 1, 0, 0.2, 0.45), ('knob shift +1', 1, 1, 0.2, 0.45), ('knob shift +2', 1, 2, 0.2, 0.45), ('knob shift +4', 1, 4, 0.2, 0.45), ('knob shift +6', 1, 6, 0.2, 0.45),
        ('knob s0.5 +2', 0.5, 2, 0.2, 0.45), ('knob s0.5 +3 (K)', 0.5, 3, 0.2, 0.45), ('knob s0.5 +4', 0.5, 4, 0.2, 0.45), ('knob s0.35 +3', 0.35, 3, 0.2, 0.45),
        ('pp thresh .01 bt .45', 1, 0, 0.01, 0.45), ('pp thresh .01 bt .2', 1, 0, 0.01, 0.2), ('pp thresh .01 bt .05', 1, 0, 0.01, 0.05), ('pp thresh .05 bt .3', 1, 0, 0.05, 0.3)]
base = {}
print('setting | split | all-images P / R (det) | GT gained / lost | dR [95% CI] | dP [95% CI]')
for nm, s, d, th, bt in rows:
    cs = {k: counts(R[k], s, d, th, bt) for k in R}; cs['full'] = np.concatenate([cs['dev'], cs['holdout']])
    if nm == 'original': base = cs
    for k in ['dev', 'holdout', 'full']:
        c = cs[k]; m = prf(c); dd = c[:, 2] - base[k][:, 2]; b = paired_bootstrap(base[k], c, B=3000)
        print(f"{nm:22s} | {k:7s} | {m['precision']:.4f} / {m['recall']:.4f} ({m['det']}) | +{int(dd[dd > 0].sum())} / {int(dd[dd < 0].sum())} | {b['recall'][0]:+.4f} [{b['recall'][1]:+.4f},{b['recall'][2]:+.4f}] | {b['precision'][0]:+.4f} [{b['precision'][1]:+.4f},{b['precision'][2]:+.4f}]", flush=True)
