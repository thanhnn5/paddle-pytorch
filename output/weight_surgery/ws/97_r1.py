"""R1 = last-layer edit logit' = s (logit - L0 + c), s 0.3, c 5. Build checkpoint; evaluate on a label file vs original.
usage: 97_r1.py <label_file> <tag> [data_dir]"""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
L0 = float(np.log(0.2 / 0.8)); S, C = 0.3, 5.0
sd0 = load_sd(); sd = {k: v.clone() for k, v in sd0.items()}
sd['head.binarize.conv3.weight'] *= S; sd['head.binarize.conv3.bias'] = S * (sd['head.binarize.conv3.bias'] - L0 + C) + L0
path = os.path.join(WS, 'cand_R1_s0.3_c5.pth'); save_sd(sd, path, sd0)
lf, tag = sys.argv[1:3]; dd = sys.argv[3] if len(sys.argv) > 3 else VAL_DIR
res = {}
for nm, s in [('orig', sd0), ('R1', sd)]:
    r = run(make_model(s), lf, dd); res[nm] = r['cnt']; res['empty'] = r['empty']
w = evalpy_weights(res['empty']); print(f'=== {tag}: {len(w)} images, {int(res["empty"].sum())} empty')
for nm in ['orig', 'R1']:
    c = res[nm]; d = c[:, 2] - res['orig'][:, 2]
    print(f"{nm:5s} all {fmt(prf(c))} | evalpy {fmt(prf(c, w))} | boxes on empty {int(c[res['empty'], 1].sum())} | gained/lost +{int(d[d > 0].sum())} / {int(d[d < 0].sum())}")
print('R1 vs orig: all   ', fmt_boot(paired_bootstrap(res['orig'], res['R1'])))
print('R1 vs orig: evalpy', fmt_boot(paired_bootstrap(res['orig'], res['R1'], w)))
pickle.dump(res, open(os.path.join(WS, f'eval_R1_{tag}.pkl'), 'wb'))
