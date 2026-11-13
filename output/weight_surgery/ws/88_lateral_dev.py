"""Lateral gains on val_dev (hooks on neck.ins_conv[k] outputs; in5 top-down route via G_td). Dose-response of the
amplifications that recovered misses on the train lab, measured on an unselected sample."""
import os, sys, pickle, types, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lf = os.path.join(WS, 'val_dev.txt'); model = make_model(); G = dict(td=1., g=[1., 1., 1., 1.])
def fwd(self, x):
    c2, c3, c4, c5 = x
    in5 = self.ins_conv[3](c5); in4 = self.ins_conv[2](c4) * G['g'][2]; in3 = self.ins_conv[1](c3) * G['g'][1]; in2 = self.ins_conv[0](c2) * G['g'][0]
    out4 = in4 + F.interpolate(in5 * G['td'], scale_factor=2, mode="nearest")
    out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")
    out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")
    p5 = F.interpolate(self._inp_forward(in5, 3), scale_factor=8, mode="nearest")
    p4 = F.interpolate(self._inp_forward(out4, 2), scale_factor=4, mode="nearest")
    p3 = F.interpolate(self._inp_forward(out3, 1), scale_factor=2, mode="nearest")
    return torch.cat([p5, p4, p3, self._inp_forward(out2, 0)], dim=1)
model.neck.forward = types.MethodType(fwd, model.neck)
cfgs = [(1, [1, 1, 1, 1]), (1, [1, 1, 1.25, 1]), (1, [1, 1, 1.5, 1]), (1, [1, 1, 2, 1]), (1, [1, 1.5, 1, 1]), (1, [1, 2, 1, 1]), (1, [2, 1, 1, 1]),
        (0.75, [1, 1, 1.25, 1]), (0.75, [1, 1, 1.5, 1]), (0.85, [1, 1, 1, 1]), (0.65, [1, 1, 1, 1])]
base = None
print('g_td [in2 in3 in4] | evalpy P / R / H | all P / R (gt det tp) | boxes on empty | GT gained / lost | paired dR, dP (all) [95% CI]')
for td, g in cfgs:
    G.update(td=td, g=g); r = run(model, lf); c = r['cnt']; w = evalpy_weights(r['empty']); a = prf(c, w); m = prf(c)
    if base is None: base = c
    d = c[:, 2] - base[:, 2]; bt = paired_bootstrap(base, c, B=2000)
    print(f"{td:4.2f} {g[:3]} | {a['precision']:.4f} / {a['recall']:.4f} / {a['hmean']:.4f} | {m['precision']:.4f} / {m['recall']:.4f} ({m['gt']} {m['det']} {m['tp']}) | {int(c[r['empty'], 1].sum())} | +{int(d[d > 0].sum())} / {int(d[d < 0].sum())} | dR {bt['recall'][0]:+.4f} [{bt['recall'][1]:+.4f},{bt['recall'][2]:+.4f}] dP {bt['precision'][0]:+.4f} [{bt['precision'][1]:+.4f},{bt['precision'][2]:+.4f}]", flush=True)
