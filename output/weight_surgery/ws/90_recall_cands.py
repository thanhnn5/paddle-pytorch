"""Recall candidates as exact weight edits, evaluated on val_dev at the fixed postprocess.
T  : in5 (stage-4 lateral) scaled by g only on its top-down route into out4/out3/out2; direct p5 branch unchanged.
     ins_conv.3.in_conv.weight *= g ; ins_conv.3.se_block.conv1.weight /= g (SE gate unchanged) ;
     the four depthwise convs of inp_conv_dw.3 /= g (their BN inputs, hence the p5 branch, unchanged).
K  : last-layer knob  logit' = s (logit - L0) + L0 + delta   (baseline-B class).
TK : both."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
L0 = float(np.log(0.2 / 0.8))
def edit_T(sd, g):
    sd = {k: v.clone() for k, v in sd.items()}
    sd['neck.ins_conv.3.in_conv.weight'] *= g; sd['neck.ins_conv.3.se_block.conv1.weight'] /= g
    for n in ['lk_origin', 'dil_conv_k5_1', 'dil_conv_k3_2', 'dil_conv_k3_3']: sd[f'neck.inp_conv_dw.3.{n}.weight'] /= g
    return sd
def edit_K(sd, s, delta):
    sd = {k: v.clone() for k, v in sd.items()}
    sd['head.binarize.conv3.weight'] *= s; sd['head.binarize.conv3.bias'] = s * (sd['head.binarize.conv3.bias'] - L0) + L0 + delta
    return sd
if __name__ == '__main__':
    sd0 = load_sd(); lf = os.path.join(WS, 'val_dev.txt')
    C = {'orig': sd0, 'T0.75': edit_T(sd0, 0.75), 'K_s0.5_d3': edit_K(sd0, 0.5, 3.), 'TK': edit_K(edit_T(sd0, 0.75), 0.5, 3.), 'T0.85': edit_T(sd0, 0.85), 'TK0.85': edit_K(edit_T(sd0, 0.85), 0.5, 3.)}
    res = {}
    print('val_dev fixed pp | evalpy P / R / H | all P / R (gt det tp) | boxes on empty | GT gained / lost | dR, dP, dH all [95% CI]')
    for nm, sd in C.items():
        if nm != 'orig': save_sd(sd, os.path.join(WS, f'cand_R_{nm}.pth'), sd0)
        r = run(make_model(sd), lf); c = r['cnt']; res[nm] = c; w = evalpy_weights(r['empty']); a = prf(c, w); m = prf(c)
        d = c[:, 2] - res['orig'][:, 2]; bt = paired_bootstrap(res['orig'], c, B=4000); be = paired_bootstrap(res['orig'], c, w, B=4000)
        print(f"{nm:10s} | {a['precision']:.4f} / {a['recall']:.4f} / {a['hmean']:.4f} | {m['precision']:.4f} / {m['recall']:.4f} ({m['gt']} {m['det']} {m['tp']}) | {int(c[r['empty'], 1].sum())} | +{int(d[d > 0].sum())} / {int(d[d < 0].sum())} | all: {fmt_boot(bt)} || evalpy: {fmt_boot(be)}", flush=True)
    pickle.dump(res, open(os.path.join(WS, 'recall_cands_dev.pkl'), 'wb'))
