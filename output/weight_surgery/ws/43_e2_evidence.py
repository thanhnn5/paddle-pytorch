"""Evidence for E2: (1) channel 79 activation in TP boxes vs FP boxes by manually tallied kind (val_dev);
(2) random-channel control on the train HeadLab: same edit shape (5 channels x0, 4 channels x2) on random channels."""
import os, sys, pickle, json, collections, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
dv = pickle.load(open(os.path.join(WS, 'boxfeat_dev.pkl'), 'rb')); tally = json.load(open(os.path.join(WS, 'fp_manual_tally_dev.json')))
r = pickle.load(open(os.path.join(WS, 'cache_logits_base_dev.pkl'), 'rb'))
kind = {}
for t in tally:
    i = t['img']; bx = postprocess((1 / (1 + np.exp(-r['logits'][i].astype(np.float32)))), r['shapes'][i], FIXED_PP)
    kind[(i, tuple(map(tuple, np.asarray(bx[t['det']]).tolist())))] = t['kind']
X = dv['X']['fuse']; rows = dv['rows']
grp = collections.defaultdict(list)
for k, row in enumerate(rows):
    if row['score'] < 0.45: continue
    if row['label'] == 'tp': grp['TP'].append(k)
    else:
        kd = kind.get((row['img'], tuple(map(tuple, row['box']))))
        if kd: grp['FP ' + kd].append(k)
allk = sum(grp.values(), []); sd_all = X[allk].std(0)
print('pooled neck-channel activation by box group (val_dev), in units of the std over all boxes')
print('group      n   ' + ' '.join(f'ch{c:2d}' for c in [79, 83, 80, 66, 88, 38, 77, 68, 50]))
for g in ['TP', 'FP N', 'FP L', 'FP X']:
    print(f'{g:6s} {len(grp[g]):5d}   ' + ' '.join(f'{X[grp[g]][:, c].mean() / sd_all[c]:+5.2f}' for c in [79, 83, 80, 66, 88, 38, 77, 68, 50]))
if '--control' in sys.argv:
    from headlab import *
    lab = HeadLab(os.path.join(WS, 'train_feat3000.txt'), list(range(0, 300)) + list(range(727, 1127)))
    def run(sup, amp):
        lab.reset(); w = lab.sd0['conv1.weight'].clone(); w[:, sup] *= 0.; w[:, amp] *= 2.; lab.head.conv1.weight.data.copy_(w); return lab.effect()
    e = run([79, 83, 80, 66, 88], [38, 77, 68, 50]); print('E2 on train lab: dTP %+.3f dFP %+.3f newbg %+d AUC %.4f (baseline AUC 0.8885)' % e)
    rng = np.random.default_rng(0); ctl = []
    for _ in range(20):
        p = rng.permutation(96); ctl.append(run(p[:5].tolist(), p[5:9].tolist()))
    ctl = np.array(ctl); print('20 random channel sets, same edit shape: AUC mean %.4f min %.4f max %.4f | dFP-dTP mean %+.3f min %+.3f max %+.3f' % (ctl[:, 3].mean(), ctl[:, 3].min(), ctl[:, 3].max(), (ctl[:, 1] - ctl[:, 0]).mean(), (ctl[:, 1] - ctl[:, 0]).min(), (ctl[:, 1] - ctl[:, 0]).max()))
