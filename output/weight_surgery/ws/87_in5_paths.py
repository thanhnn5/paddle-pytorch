"""H8 decomposition: in5 (stage-4 lateral) reaches the head by two routes: top-down (added into out4 -> out3 -> out2)
and the direct p5 branch. Gains g_td and g_p5 on each route separately, by overriding the neck forward (experiment only).
usage: 87_in5_paths.py <label_file> <tag>"""
import os, sys, pickle, types, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lf, tag = sys.argv[1:3]
model = make_model(); G = dict(td=1., p5=1.)
def fwd(self, x):
    c2, c3, c4, c5 = x
    in5 = self.ins_conv[3](c5); in4 = self.ins_conv[2](c4); in3 = self.ins_conv[1](c3); in2 = self.ins_conv[0](c2)
    out4 = in4 + F.interpolate(in5 * G['td'], scale_factor=2, mode="nearest")
    out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")
    out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")
    p5 = F.interpolate(self._inp_forward(in5 * G['p5'], 3), scale_factor=8, mode="nearest")
    p4 = F.interpolate(self._inp_forward(out4, 2), scale_factor=4, mode="nearest")
    p3 = F.interpolate(self._inp_forward(out3, 1), scale_factor=2, mode="nearest")
    return torch.cat([p5, p4, p3, self._inp_forward(out2, 0)], dim=1)
model.neck.forward = types.MethodType(fwd, model.neck)
cfgs = [(1, 1), (0.75, 1), (0.5, 1), (0.25, 1), (0, 1), (1, 0.75), (1, 0.5), (1, 0.25), (1, 0), (0.75, 0.75), (0.5, 0.5), (0.25, 0.25), (0, 0), (0.5, 0), (0, 0.5)]
res = {}; base = None
print(f'{tag}: g_td g_p5 | evalpy P / R / H | all P / R (gt det tp) | boxes on empty | GT gained / lost | paired dR, dP (all) [95% CI]')
for td, p5 in cfgs:
    G.update(td=td, p5=p5); r = run(model, lf); c = r['cnt']; res[(td, p5)] = c; w = evalpy_weights(r['empty']); a = prf(c, w); m = prf(c)
    if base is None: base = c
    d = c[:, 2] - base[:, 2]; bt = paired_bootstrap(base, c, B=2000)
    print(f"{td:4.2f} {p5:4.2f} | {a['precision']:.4f} / {a['recall']:.4f} / {a['hmean']:.4f} | {m['precision']:.4f} / {m['recall']:.4f} ({m['gt']} {m['det']} {m['tp']}) | {int(c[r['empty'], 1].sum())} | +{int(d[d > 0].sum())} / {int(d[d < 0].sum())} | dR {bt['recall'][0]:+.4f} [{bt['recall'][1]:+.4f},{bt['recall'][2]:+.4f}] dP {bt['precision'][0]:+.4f} [{bt['precision'][1]:+.4f},{bt['precision'][2]:+.4f}]", flush=True)
pickle.dump(res, open(os.path.join(WS, f'in5_paths_{tag}.pkl'), 'wb'))
