"""Holdout evaluation at pre-declared settings only. usage: 60_holdout.py <label_file> <tag>"""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lf, tag = sys.argv[1:3]
dd = sys.argv[3] if len(sys.argv) > 3 else VAL_DIR
FIX = FIXED_PP; AST = dict(FIXED_PP, thresh=0.1, box_thresh=0.65, unclip_ratio=1.3)
OWN = {'B': dict(FIXED_PP, thresh=0.1, box_thresh=0.5, unclip_ratio=1.3), 'BE2': dict(FIXED_PP, thresh=0.1, box_thresh=0.4, unclip_ratio=1.4)}
CK = {'orig': ORIG, 'E2': WS + '/cand_E2_sup5x0_amp4x2.pth', 'B': WS + '/cand_B_s0.25.pth', 'BE2': WS + '/cand_BE2_s0.25_sup5x0_amp4x2.pth'}
which = sys.argv[4].split(',') if len(sys.argv) > 4 else list(CK)
res = {}
for nm in which:
    model = make_model(load_sd(CK[nm])); L = {}
    model.head.binarize.conv3.register_forward_hook(lambda m, i, o: L.__setitem__('l', o[0, 0].float().cpu().numpy()))
    pps = {'fixed': FIX, 'A*': AST}
    if nm in OWN: pps['own'] = OWN[nm]
    cnts = {k: [] for k in pps}; posts = {k: DBPostProcess(**v) for k, v in pps.items()}
    def hook(i, rec):
        prob = 1 / (1 + np.exp(-L['l']))
        for k in pps: cnts[k].append(score_image(postprocess(prob, rec['shape'], posts[k]), rec['polys'], rec['tags']))
    r = run(model, lf, dd, hook=hook)
    res[nm] = {k: np.array(v) for k, v in cnts.items()}; res['empty'] = r['empty']
    del model
w = evalpy_weights(res['empty'])
pickle.dump(res, open(os.path.join(WS, f'eval_{tag}.pkl'), 'wb'))
print(f'=== {tag}: {len(w)} images, {int(res["empty"].sum())} empty')
for nm in which:
    for k, c in res[nm].items():
        ref = res['orig'].get(k if k != 'own' else 'A*') if 'orig' in res else None
        print(f'{nm:5s} {k:6s} evalpy {fmt(prf(c, w))} | all {fmt(prf(c))}')
        if nm != 'orig' and ref is not None:
            lab = 'same pp' if k != 'own' else 'A*'
            print(f'             vs orig@{lab}: evalpy {fmt_boot(paired_bootstrap(ref, c, w))}\n             vs orig@{lab}: all    {fmt_boot(paired_bootstrap(ref, c))}')
if 'B' in res and 'BE2' in res:
    for k in ['fixed', 'own']:
        print(f'BE2 vs B @{k}: evalpy {fmt_boot(paired_bootstrap(res["B"][k], res["BE2"][k], w))} | all {fmt_boot(paired_bootstrap(res["B"][k], res["BE2"][k]))}')
