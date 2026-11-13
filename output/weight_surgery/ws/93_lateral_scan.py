"""Channel-level causal scan of the neck laterals for recall. For each of the 4 x 96 lateral channels (output of
neck.ins_conv[k]) apply gain 0 and 2 and re-run neck + head on cached backbone features of train lab images.
Reports the logit change inside fully missed GT, in TP / FP boxes, and far-background pixels above the contour."""
import os, sys, pickle, types, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from fnlab import *
ix = pickle.load(open(os.path.join(WS, 'fnlab_train_idx.pkl'), 'rb')); rng = np.random.default_rng(1)
pos = np.flatnonzero(np.array(ix['ctrl'])); ctrl = set(rng.choice(pos, 220, replace=False).tolist())
lab = FNLab(os.path.join(WS, 'fnlab_train.txt'), ctrl, keep_bb=True)
for it in lab.items: it['fuse'] = None
neck = lab.model.neck; GV = [torch.ones(1, 96, 1, 1, device='cuda') for _ in range(4)]
def neck_fwd(bb):
    c2, c3, c4, c5 = [x.float() for x in bb]
    in5 = neck.ins_conv[3](c5) * GV[3]; in4 = neck.ins_conv[2](c4) * GV[2]; in3 = neck.ins_conv[1](c3) * GV[1]; in2 = neck.ins_conv[0](c2) * GV[0]
    out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest"); out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest"); out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")
    return torch.cat([F.interpolate(neck._inp_forward(in5, 3), scale_factor=8, mode="nearest"), F.interpolate(neck._inp_forward(out4, 2), scale_factor=4, mode="nearest"),
                      F.interpolate(neck._inp_forward(out3, 1), scale_factor=2, mode="nearest"), neck._inp_forward(out2, 0)], dim=1)
h = lab.head
lab.logits = lambda it: h.conv3(F.relu(h.conv_bn2(h.conv2(F.relu(h.conv_bn1(h.conv1(neck_fwd(it['bb'])))))))) [0, 0].flatten()
lab.base = lab.measure(); res = {}
for k in range(4):
    for c in range(96):
        for g in (0., 2.):
            GV[k][0, c] = g; res[(k, c, g)] = lab.effect(); GV[k][0, c] = 1.
    print('level', k, 'done', flush=True)
pickle.dump(res, open(os.path.join(WS, 'lateral_scan_train.pkl'), 'wb'))
nm = ['in2', 'in3', 'in4', 'in5']
for g in (0., 2.):
    print(f'--- gain {g}: top 20 by dMiss | lateral ch | dMiss rec | dTP dFP | far-bg px')
    for key in sorted([x for x in res if x[2] == g], key=lambda x: -res[x]['dmiss'])[:20]:
        e = res[key]; print(f"  {nm[key[0]]} {key[1]:2d} | {e['dmiss']:+.2f} {e['rec']:.3f} | {e['dtp']:+.2f} {e['dfp']:+.2f} | {e['dbg']:+d}")
    print(f'--- gain {g}: selective (dMiss >= 0.15, bg px <= +300, dTP >= -0.15), by dMiss')
    for key in sorted([x for x in res if x[2] == g and res[x]['dmiss'] >= 0.15 and res[x]['dbg'] <= 300 and res[x]['dtp'] >= -0.15], key=lambda x: -res[x]['dmiss'])[:30]:
        e = res[key]; print(f"  {nm[key[0]]} {key[1]:2d} | {e['dmiss']:+.2f} {e['rec']:.3f} | {e['dtp']:+.2f} {e['dfp']:+.2f} | {e['dbg']:+d}")
