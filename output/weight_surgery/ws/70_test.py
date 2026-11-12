"""FINAL TEST. Original and the frozen checkpoint, once per resize protocol, at frozen settings only."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
TD = os.path.join(ROOT, 'data/ocr_test_20251127/images'); TL = os.path.join(ROOT, 'data/ocr_test_20251127/test.txt')
CK = {'orig': ORIG, 'B': os.path.join(WS, 'best_edited.pth')}
PROT = {'default': (None, {'fixed': FIXED_PP, 'A*': dict(FIXED_PP, thresh=0.1, box_thresh=0.65, unclip_ratio=1.3)}),
        '1280max': ({'limit_side_len': 1280, 'limit_type': 'max'}, {'bt0.6': dict(FIXED_PP, box_thresh=0.6)})}
res = {}
for pn, (rk, pps) in PROT.items():
    for nm, ck in CK.items():
        model = make_model(load_sd(ck)); posts = {k: DBPostProcess(**v) for k, v in pps.items()}; cnts = {k: [] for k in pps}; P = {}
        hk = model.head.register_forward_hook(lambda m, i, o: P.__setitem__('p', o['res'][0, 0].float().cpu().numpy()))
        def hook(i, rec):
            for k in pps: cnts[k].append(score_image(postprocess(P['p'], rec['shape'], posts[k]), rec['polys'], rec['tags']))
        r = run(model, TL, TD, resize_kwargs=rk, hook=hook)
        for k in pps: res[(pn, nm, k)] = np.array(cnts[k])
        res['empty'] = r['empty']; del model
pickle.dump(res, open(os.path.join(WS, 'eval_test.pkl'), 'wb'))
w = evalpy_weights(res['empty']); e = res['empty']
print(f'test: {len(e)} images, {int(e.sum())} empty')
for key in [k for k in res if k != 'empty']:
    c = res[key]
    print(f'{str(key):32s} all {fmt(prf(c))} | evalpy {fmt(prf(c, w))} | boxes on empty images {int(c[e, 1].sum())} (images {int((c[e, 1] > 0).sum())})')
def cmp(a, b, lab):
    print(f'{lab}\n    all    {fmt_boot(paired_bootstrap(res[a], res[b]))}\n    evalpy {fmt_boot(paired_bootstrap(res[a], res[b], w))}')
cmp(('default', 'orig', 'fixed'), ('default', 'B', 'fixed'), 'B@fixed vs orig@fixed')
cmp(('default', 'orig', 'fixed'), ('default', 'orig', 'A*'), 'orig@A* vs orig@fixed (postprocess alone)')
cmp(('default', 'orig', 'A*'), ('default', 'B', 'fixed'), 'B@fixed vs orig@A*')
cmp(('default', 'orig', 'A*'), ('default', 'B', 'A*'), 'B@A* vs orig@A*')
cmp(('1280max', 'orig', 'bt0.6'), ('1280max', 'B', 'bt0.6'), 'secondary: B vs orig at 1280-max, box_thresh 0.6')
