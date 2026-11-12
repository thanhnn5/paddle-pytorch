"""Candidates on val_dev: postprocess sweep on top of each (A combined with the edit), paired bootstrap,
and which manually-tallied FP kinds (N unannotated number / L letters / X non-text) each candidate removes at fixed pp."""
import os, sys, pickle, json, collections, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sweep import *
from taxonomy import poly
TH = [0.1, 0.2, 0.3]; BT = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]; UC = [1.3, 1.4, 1.5]
ASTAR = (0.1, 0.65, 1.3)
names = {'orig': 'cache_logits_base_dev.pkl', 'B': 'cache_logits_cand_B_s0.25_dev.pkl', 'E2': 'cache_logits_cand_E2_sup5x0_amp4x2_dev.pkl', 'BE2': 'cache_logits_cand_BE2_s0.25_sup5x0_amp4x2_dev.pkl'}
tally = json.load(open(os.path.join(WS, 'fp_manual_tally_dev.json')))
out = {}; boxes0 = None
for nm, f in names.items():
    r = pickle.load(open(os.path.join(WS, f), 'rb')); w = evalpy_weights(r['empty'])
    r['probs'] = [(1 / (1 + np.exp(-l.astype(np.float32)))) for l in r['logits']]; r['logits'] = None
    res, _ = sweep(r, TH, BT, UC); out[nm] = res
    fx = res[:, 1, 3, 1]; a = res[:, 0, BT.index(0.65), 0]
    g = max(((prf(res[:, ti, bi, ui], w)['hmean'], TH[ti], BT[bi], UC[ui]) for ti in range(3) for bi in range(len(BT)) for ui in range(3)))
    ga = max(((prf(res[:, ti, bi, ui])['hmean'], TH[ti], BT[bi], UC[ui]) for ti in range(3) for bi in range(len(BT)) for ui in range(3)))
    print(f"{nm:5s} fixed pp : evalpy {fmt(prf(fx, w))} | all {fmt(prf(fx))}")
    print(f"      at A* {ASTAR}: evalpy {fmt(prf(a, w))} | all {fmt(prf(a))}")
    print(f"      own best pp on dev: evalpy {g[0]:.4f} @ {g[1:]} | all {ga[0]:.4f} @ {ga[1:]}")
    if nm != 'orig':
        print('      vs orig fixed pp  evalpy', fmt_boot(paired_bootstrap(out['orig'][:, 1, 3, 1], fx, w)), '\n                        all   ', fmt_boot(paired_bootstrap(out['orig'][:, 1, 3, 1], fx)))
        print('      vs orig, both at A*  evalpy', fmt_boot(paired_bootstrap(out['orig'][:, 0, BT.index(0.65), 0], a, w)))
    if nm == 'BE2':
        print('      BE2 vs B, fixed pp  evalpy', fmt_boot(paired_bootstrap(out['B'][:, 1, 3, 1], fx, w)), '\n                          all   ', fmt_boot(paired_bootstrap(out['B'][:, 1, 3, 1], fx)))
        for (t, b, u) in [(0.1, 0.5, 1.3), (0.2, 0.5, 1.4), (0.1, 0.45, 1.3)]:
            i = (TH.index(t), BT.index(b), UC.index(u)); print(f'      BE2 vs B, both at {(t, b, u)} evalpy', fmt_boot(paired_bootstrap(out['B'][:, i[0], i[1], i[2]], res[:, i[0], i[1], i[2]], w)))
    # FP kinds surviving at fixed pp
    bx = [postprocess(r['probs'][i].astype(np.float32), r['shapes'][i], FIXED_PP) for i in range(len(r['probs']))]
    if boxes0 is None: boxes0 = bx
    surv = collections.Counter(); tot = collections.Counter()
    for t in tally:
        b0 = poly(boxes0[t['img']][t['det']]); tot[t['kind']] += 1
        if any(poly(b).intersection(b0).area / max(poly(b).union(b0).area, 1e-9) > 0.5 for b in bx[t['img']]): surv[t['kind']] += 1
    print('      of the 186 tallied baseline FPs still present: ' + ', '.join(f"{k} {surv[k]}/{tot[k]}" for k in 'NLX'), flush=True)
pickle.dump(out, open(os.path.join(WS, 'cand_sweeps_dev.pkl'), 'wb'))
