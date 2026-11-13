"""What are the extra boxes? For several edits (neck gains via forward override, last-layer shift), list the boxes
that are new (no baseline box overlaps them) and not matched to GT, and render crop sheets for a by-eye tally.
usage: 92_new_boxes.py <label_file> <tag> [configs]"""
import os, sys, pickle, types, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from taxonomy import analyse_image, crop_sheet, poly
lf, tag = sys.argv[1:3]
model = make_model(); G = dict(td=1., p5=1., g=[1., 1., 1.], shift=0.)
def fwd(self, x):
    c2, c3, c4, c5 = x
    in5 = self.ins_conv[3](c5); in4 = self.ins_conv[2](c4) * G['g'][2]; in3 = self.ins_conv[1](c3) * G['g'][1]; in2 = self.ins_conv[0](c2) * G['g'][0]
    out4 = in4 + F.interpolate(in5 * G['td'], scale_factor=2, mode="nearest")
    out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")
    out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")
    p5 = F.interpolate(self._inp_forward(in5 * G['p5'], 3), scale_factor=8, mode="nearest")
    p4 = F.interpolate(self._inp_forward(out4, 2), scale_factor=4, mode="nearest")
    p3 = F.interpolate(self._inp_forward(out3, 1), scale_factor=2, mode="nearest")
    return torch.cat([p5, p4, p3, self._inp_forward(out2, 0)], dim=1)
model.neck.forward = types.MethodType(fwd, model.neck)
b0 = model.head.binarize.conv3.bias.data.clone()
CF = {'base': {}, 'in4x1.5': dict(g=[1, 1, 1.5]), 'in4x2': dict(g=[1, 1, 2.]), 'td0': dict(td=0.), 'td0.5_in4x1.5': dict(td=0.5, g=[1, 1, 1.5]),
      'in3x2': dict(g=[1, 2., 1]), 'shift6': dict(shift=6.), 'in2x2': dict(g=[2., 1, 1])}
which = sys.argv[3].split(',') if len(sys.argv) > 3 else list(CF)
out = {}; base = None; rng = np.random.default_rng(0)
os.makedirs(os.path.join(WS, 'viz', 'new_' + tag), exist_ok=True)
for nm in ['base'] + [w for w in which if w != 'base']:
    G.update(td=1., p5=1., g=[1., 1., 1.], shift=0.); G.update(CF[nm]); model.head.binarize.conv3.bias.data.copy_(b0 + G['shift'])
    r = run(model, lf, keep_boxes=True)
    if nm == 'base': base = r; continue
    items = []; gained = lost = 0
    for i in range(len(r['names'])):
        polys, tags = r['gts'][i]; dets, gts, _ = analyse_image(r['boxes'][i], polys, tags); un = {d['d'] for d in dets}
        d = r['cnt'][i, 2] - base['cnt'][i, 2]; gained += max(d, 0); lost += max(-d, 0)
        B0 = [poly(b) for b in base['boxes'][i]]
        for d_ in un:
            q = poly(r['boxes'][i][d_])
            if q.area > 0 and all(q.intersection(b).area < 0.2 * min(q.area, b.area) for b in B0 if b.area > 0): items.append((i, r['boxes'][i][d_], f'{i}'))
    m = prf(r['cnt']); mb = prf(base['cnt'])
    print(f"{nm:14s} all P {m['precision']:.4f} R {m['recall']:.4f} det {m['det']} (base {mb['det']}) | GT gained/lost +{gained}/-{lost} | new unmatched boxes {len(items)}", flush=True)
    out[nm] = [(i, np.asarray(b).tolist()) for i, b, _ in items]
    sel = rng.permutation(len(items))[:60]; it = [items[k] for k in sel]
    r2 = dict(r); r2['boxes'] = [[b for (j, b, _) in it if j == i] for i in range(len(r['names']))]
    for s in range(0, len(it), 30): crop_sheet(r2, VAL_DIR, it[s:s + 30], os.path.join(WS, 'viz', 'new_' + tag, f'{nm}_{s // 30}.jpg'), cols=6, maxn=30, pad=1.0)
pickle.dump(out, open(os.path.join(WS, f'new_boxes_{tag}.pkl'), 'wb'))
