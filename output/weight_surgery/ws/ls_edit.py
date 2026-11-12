"""Closed-form constrained least-squares edit of head.binarize.conv3 (final layer of the binarize head).
  minimise  sum_keep w_s (dw.x)^2 + kappa * sum_{FP active} (dw.x + Delta)^2 + lam |dw|^2 ,  x = [h2, 1]
  => dw = -Delta * kappa * (M + lam I)^-1 s_fp ;  the same dw is added to all four taps of the 2x2 transposed conv.
Statistics come from train images only (30_pixstats.py)."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *

def solve(st, kappa=3.0, w_bg_low=0.01, w_tp=1.0, lam_rel=1e-3, fp_bins=(2, 3, 4, 5), w_fp_low=0.0, w_miss=1.0):
    C, N = st['C'], st['N']
    Cfp = sum(C[0, b] for b in fp_bins); nfp = sum(N[0, b] for b in fp_bins)
    M = kappa * Cfp
    for b in range(6):
        if b not in fp_bins: M = M + w_fp_low * C[0, b]
        M = M + w_tp * C[1, b] + w_miss * C[2, b] + (w_bg_low if b == 0 else 1.0) * C[3, b]
    M = M + lam_rel * np.trace(M) / 25 * np.eye(25)
    dw = -kappa * np.linalg.solve(M, Cfp[:, 24])     # for Delta = 1
    def stat(Cs, n):  # mean and rms of dw.x over a stratum
        return float(dw @ Cs[:, 24] / n), float(np.sqrt(dw @ Cs @ dw / n))
    rep = dict(fp_active=stat(Cfp, nfp),
               tp_active=stat(sum(C[1, b] for b in (2, 3, 4, 5)), sum(N[1, b] for b in (2, 3, 4, 5))),
               tp_border=stat(C[1, 1] + C[1, 2], N[1, 1] + N[1, 2]),
               tp_low=stat(C[1, 0], N[1, 0]), bg_low=stat(C[3, 0], N[3, 0]),
               bg_mid=stat(sum(C[3, b] for b in (1, 2, 3)), sum(N[3, b] for b in (1, 2, 3))))
    return dw, rep

def apply(sd0, dw, delta):
    sd = {k: v.clone() for k, v in sd0.items()}
    sd['head.binarize.conv3.weight'] += delta * torch.tensor(dw[:24], dtype=torch.float32)[:, None, None, None]
    sd['head.binarize.conv3.bias'] += float(delta * dw[24])
    return sd

if __name__ == '__main__':
    st = pickle.load(open(os.path.join(WS, 'pixstats_train3000.pkl'), 'rb'))
    print('per unit Delta: mean (rms) logit change by region')
    for kappa in [1, 3, 10, 30]:
        for wbg in [0.001, 0.01, 0.1]:
            dw, r = solve(st, kappa, wbg)
            print(f'kappa {kappa:4.0f} w_bg_low {wbg:5.3f} | ' + ' | '.join(f'{k} {v[0]:+.3f} ({v[1]:.3f})' for k, v in r.items()) + f' | |dw| {np.linalg.norm(dw[:24]):.3f} db {dw[24]:+.3f}')
