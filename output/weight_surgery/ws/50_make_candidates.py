"""Build the candidate checkpoints.
 B   : conv3 scaled by s=0.25 about L0=logit(0.2)                       [global knob, baseline B]
 E2  : head.binarize.conv1 reads neck channels 79,83,80,66,88 with gain 0 and 38,77,68,50 with gain 2   [targeted]
 BE2 : both."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
L0 = float(np.log(0.2 / 0.8)); S = 0.25; SUP = [79, 83, 80, 66, 88]; AMP = [38, 77, 68, 50]
sd0 = load_sd()
def editB(sd, s=S):
    sd = {k: v.clone() for k, v in sd.items()}
    sd['head.binarize.conv3.weight'] *= s
    sd['head.binarize.conv3.bias'] = s * (sd['head.binarize.conv3.bias'] - L0) + L0
    return sd
def editE2(sd):
    sd = {k: v.clone() for k, v in sd.items()}
    for c in SUP: sd['head.binarize.conv1.weight'][:, c] *= 0.0
    for c in AMP: sd['head.binarize.conv1.weight'][:, c] *= 2.0
    return sd
if __name__ == '__main__':
    for nm, sd in [('cand_B_s0.25', editB(sd0)), ('cand_E2_sup5x0_amp4x2', editE2(sd0)), ('cand_BE2_s0.25_sup5x0_amp4x2', editB(editE2(sd0)))]:
        save_sd(sd, os.path.join(WS, nm + '.pth'), sd0)
        ch = [(k, float((sd[k] - sd0[k]).norm() / sd0[k].norm())) for k in sd0 if not torch.equal(sd[k], sd0[k])]
        print(nm, 'changed tensors (relative L2 change):', ch)
