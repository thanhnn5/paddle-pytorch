"""H1: BatchNorm running stats vs activations measured on train images under the eval pipeline.
Measures (eval mode, original stats upstream) mean/var of every BN input over pixels and images."""
import os, sys, numpy as np, torch, pickle
sys.path.insert(0, os.path.dirname(__file__))
from common import *
N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 800
TRAIN = os.path.join(VAL_DIR, 'train.txt')
# deterministic train subset
items = read_label_file(TRAIN)
rng = np.random.default_rng(0)
sel = sorted(rng.permutation(len(items))[:N_IMG].tolist())
sub = os.path.join(WS, f'train_sub{N_IMG}.txt')
open(sub, 'w').write('\n'.join(items[i][0] + '\t' + items[i][1] for i in sel) + '\n')

model = make_model()
bns = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
print('BN layers:', len(bns), 'aux:', sum('aux' in n for n, _ in bns), 'thresh:', sum('head.thresh' in n for n, _ in bns))
acc = {n: [0., 0., 0] for n, _ in bns}  # per-image mean of mean, mean of E[x^2], count
order = []
def mk(n):
    def h(m, inp, out):
        x = inp[0].double()
        if n not in order: order.append(n)
        a = acc[n]
        a[0] = a[0] + x.mean((0, 2, 3)); a[1] = a[1] + (x * x).mean((0, 2, 3)); a[2] += 1
    return h
hs = [m.register_forward_hook(mk(n)) for n, m in bns]
r = run(model, sub)
print('train subset (eval pipeline):', fmt(prf(r['cnt'])), '| evalpy', fmt(prf(r['cnt'], evalpy_weights(r['empty']))))
for h in hs: h.remove()
sd = model.state_dict(); rows = []; stats = {}
for n in order:
    a = acc[n]
    mu = (a[0] / a[2]).float().cpu(); var = (a[1] / a[2]).float().cpu() - mu ** 2
    rm, rv = sd[n + '.running_mean'].cpu(), sd[n + '.running_var'].cpu()
    z = (mu - rm) / (rv + 1e-5).sqrt(); lr = torch.log2((var + 1e-5) / (rv + 1e-5))
    stats[n] = (mu, var)
    rows.append((n, float(z.abs().mean()), float(z.abs().max()), float(lr.mean()), float(lr.abs().max())))
pickle.dump(dict(order=order, stats=stats), open(os.path.join(WS, f'bn_stats_train{N_IMG}.pkl'), 'wb'))
print('executed BN layers (eval forward):', len(order), ' never executed:', [n for n, _ in bns if n not in order][:4], '...')
print(f"{'layer':58s} mean|z| max|z| mean log2(var_meas/var_run) max|log2|")
for r_ in rows: print(f'{r_[0]:58s} {r_[1]:.3f} {r_[2]:.3f} {r_[3]:+.3f} {r_[4]:.3f}')
allz = np.array([r_[1] for r_ in rows]); alll = np.array([r_[3] for r_ in rows])
print('overall: mean|z| %.3f, mean log2 var ratio %+.3f' % (allz.mean(), alll.mean()))
