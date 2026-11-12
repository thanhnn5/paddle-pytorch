"""Anatomy of the TP-vs-FP direction in h2 (input of head.binarize.conv3) and confound checks."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
exec(open(os.path.join(os.path.dirname(__file__), '21_linear_sep.py')).read().split("print('train boxes'")[0])
Xt, Xd = tr['X']['h2'][kt], dv['X']['h2'][kd]
w, b = lda(Xt, yt, 0.1)
sd = load_sd(); W3 = sd['head.binarize.conv3.weight'][:, 0]  # 24 x 2 x 2
wm = W3.mean((1, 2)).numpy()
m1, m0 = Xt[yt == 1].mean(0), Xt[yt == 0].mean(0); s = Xt.std(0)
print('conv3 bias', float(sd['head.binarize.conv3.bias']), '| mean logit TP %.2f FP %.2f' % (arr(tr, 'logit')[kt][yt == 1].mean(), arr(tr, 'logit')[kt][yt == 0].mean()))
print('ch  conv3_w(mean of 4 taps)  mean_TP  mean_FP  d=(TP-FP)/std  current contribution to (TP-FP) logit gap  LDA w  LDA contribution to gap')
for c in np.argsort(-np.abs(w * (m1 - m0))):
    print(f'{c:2d}  {wm[c]:+7.3f}   {m1[c]:6.3f}  {m0[c]:6.3f}  {(m1[c] - m0[c]) / s[c]:+5.2f}   {wm[c] * (m1[c] - m0[c]):+6.3f}   {w[c]:+7.3f}  {w[c] * (m1[c] - m0[c]):+6.3f}')
print('current readout gap (sum w3*(m1-m0)) = %.3f ; LDA gap = %.3f' % (float(wm @ (m1 - m0)), float(w @ (m1 - m0))))
# confounds: box area, score
at, ad = np.log(arr(tr, 'area')[kt]), np.log(arr(dv, 'area')[kd])
sc = Xd @ w + b
print('AUC of log(area) alone on dev: %.3f ; corr(LDA score, log area) dev: %.3f ; corr(LDA score, model logit): %.3f' % (auc(ad, yd), np.corrcoef(sc, ad)[0, 1], np.corrcoef(sc, arr(dv, 'logit')[kd])[0, 1]))
# LDA after regressing out log-area and logit from the features (train fit)
for nm, cov_t, cov_d in [('log area', at, ad), ('model logit', arr(tr, 'logit')[kt], arr(dv, 'logit')[kd])]:
    A = np.c_[np.ones(len(cov_t)), cov_t]; beta = np.linalg.lstsq(A, Xt, rcond=None)[0]
    Rt = Xt - A @ beta; Rd = Xd - np.c_[np.ones(len(cov_d)), cov_d] @ beta
    w2, b2 = lda(Rt, yt, 0.1); print(f'LDA on h2 with {nm} regressed out: dev AUC {auc(Rd @ w2 + b2, yd):.3f}')
# size-stratified AUC
for lo, hi in [(0, 33), (33, 66), (66, 100)]:
    a0, a1 = np.percentile(ad, lo), np.percentile(ad, hi); m = (ad >= a0) & (ad <= a1)
    print(f'area pct {lo}-{hi}: n {m.sum()} tp {yd[m].sum()} | AUC score {auc(sdv[kd][m], yd[m]):.3f} | AUC LDA h2 {auc(sc[m], yd[m]):.3f}')
