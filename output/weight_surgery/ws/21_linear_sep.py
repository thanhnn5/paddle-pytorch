"""H3 diagnostic: is the TP-vs-FP distinction linearly available in the features the head reads?
Closed-form LDA (shrinkage) fitted on TRAIN boxes, AUC on val_dev boxes; compared with the model's own box score."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import WS
tr = pickle.load(open(os.path.join(WS, 'boxfeat_train3000.pkl'), 'rb')); dv = pickle.load(open(os.path.join(WS, 'boxfeat_dev.pkl'), 'rb'))
def arr(d, k): return np.array([r[k] for r in d['rows']])
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def lda(X, y, lam=0.1):
    m1, m0 = X[y == 1].mean(0), X[y == 0].mean(0)
    Xc = np.vstack([X[y == 1] - m1, X[y == 0] - m0]); S = Xc.T @ Xc / len(Xc)
    S = S + lam * np.trace(S) / len(S) * np.eye(len(S))
    w = np.linalg.solve(S, m1 - m0); return w, -(w @ (m1 + m0)) / 2
lt, ld = arr(tr, 'label'), arr(dv, 'label'); st, sdv = arr(tr, 'score'), arr(dv, 'score')
kt = (st >= 0.45) & (lt != 'other'); kd = (sdv >= 0.45) & (ld != 'other')
yt = (lt[kt] == 'tp').astype(int); yd = (ld[kd] == 'tp').astype(int)
print('train boxes', kt.sum(), 'tp', yt.sum(), '| dev boxes', kd.sum(), 'tp', yd.sum())
sub = {'all FP': np.ones(kd.sum(), bool), 'FP on non-empty imgs': ld[kd] != 'fp_empty', 'FP on empty imgs': ld[kd] != 'fp_bg'}
def report(name, s_dev):
    print(f'{name:44s}', ' | '.join(f'{k}: AUC {auc(s_dev[m], yd[m]):.3f}' for k, m in sub.items()))
report('model box score', sdv[kd]); report('model mean logit', arr(dv, 'logit')[kd])
res = {}
for feat in ['fuse', 'h1', 'h2']:
    Xt, Xd = tr['X'][feat][kt], dv['X'][feat][kd]
    for lam in [0.01, 0.1, 1.0]:
        w, b = lda(Xt, yt, lam)
        report(f'LDA on {feat} (lam {lam}), fit on train', Xd @ w + b)
    # add the model's own logit as an extra feature: does the feature add anything beyond the score?
    Xt2 = np.hstack([Xt, arr(tr, 'logit')[kt, None]]); Xd2 = np.hstack([Xd, arr(dv, 'logit')[kd, None]])
    w, b = lda(Xt2, yt, 0.1); report(f'LDA on {feat} + logit', Xd2 @ w + b)
    w, b = lda(Xt, yt, 0.1); res[feat] = (w, b)
    print(f'   train AUC (in-sample) {auc(Xt @ w + b, yt):.3f}; score AUC on train {auc(st[kt], yt):.3f}')
# fit using ONLY the never-trained empty train images as negatives
for feat in ['fuse', 'h2']:
    m = lt[kt] != 'fp_bg'
    w, b = lda(tr['X'][feat][kt][m], yt[m], 0.1); report(f'LDA on {feat}, negatives = train empty only', dv['X'][feat][kd] @ w + b)
pickle.dump(res, open(os.path.join(WS, 'lda_dirs.pkl'), 'wb'))
# what would a perfect use of the LDA score buy? operating curve on dev boxes: drop boxes with lda score < t
for feat in ['fuse', 'h2']:
    w, b = res[feat]; s = dv['X'][feat][kd] @ w + b
    print(f'-- {feat}: dropping dev boxes below LDA quantile q -> (TP kept, FP kept); vs dropping by box score')
    for q in [0.05, 0.1, 0.15, 0.2, 0.3]:
        t = np.quantile(s, q); t2 = np.quantile(sdv[kd], q)
        print(f'   q={q}: LDA tp {int(((s >= t) & (yd == 1)).sum())}/{yd.sum()} fp {int(((s >= t) & (yd == 0)).sum())}/{(1 - yd).sum()}'
              f' | score tp {int(((sdv[kd] >= t2) & (yd == 1)).sum())} fp {int(((sdv[kd] >= t2) & (yd == 0)).sum())}')
