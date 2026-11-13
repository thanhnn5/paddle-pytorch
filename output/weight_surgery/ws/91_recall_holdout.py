"""Holdout evaluation of the recall candidates T and K at the fixed postprocess (pre-declared, section 21).
Also, on val_dev only: which of the diagnosed misses each candidate recovers."""
import os, sys, pickle, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image
CK = {'orig': ORIG, 'T': WS + '/cand_R_T0.75.pth', 'K': WS + '/cand_R_K_s0.5_d3.pth'}
def go(lf, tag, keep=False):
    res = {}; un = {}
    for nm, ck in CK.items():
        r = run(make_model(load_sd(ck)), lf, keep_boxes=keep); res[nm] = r['cnt']; res['empty'] = r['empty']
        if keep: un[nm] = {(i, g['g']) for i in range(len(r['names'])) for g in analyse_image(r['boxes'][i], *r['gts'][i])[1]}
    w = evalpy_weights(res['empty']); print(f'=== {tag}: {len(w)} images, {int(res["empty"].sum())} empty')
    for nm in CK:
        c = res[nm]; d = c[:, 2] - res['orig'][:, 2]
        print(f"{nm:5s} evalpy {fmt(prf(c, w))} | all {fmt(prf(c))} | boxes on empty {int(c[res['empty'], 1].sum())} | gained/lost +{int(d[d > 0].sum())} / {int(d[d < 0].sum())}")
        if nm != 'orig': print(f"      vs orig: all {fmt_boot(paired_bootstrap(res['orig'], c))}\n               evalpy {fmt_boot(paired_bootstrap(res['orig'], c, w))}")
    pickle.dump(res, open(os.path.join(WS, f'eval_recall_{tag}.pkl'), 'wb'))
    return un
if sys.argv[1] == 'dev':
    un = go(os.path.join(WS, 'val_dev.txt'), 'val_dev', keep=True)
    why = {(x['i'], x['g']): x['why'] for x in json.load(open(os.path.join(WS, 'fn_diag_dev.json')))['fn']}
    for nm in ['T', 'K']: print(nm, 'recovers:', sorted((i, g, why.get((i, g), '?')) for i, g in un['orig'] - un[nm]), '| newly unmatched:', sorted(un[nm] - un['orig']))
else:
    go(os.path.join(WS, 'val_holdout.txt'), 'val_holdout')
