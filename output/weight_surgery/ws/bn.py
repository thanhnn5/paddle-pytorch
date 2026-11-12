"""BatchNorm statistics: forward-only measurement and recalibration."""
import os, sys, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *


def bn_names(model, executed_only=True):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)
            and not (executed_only and ('aux_' in n or 'head.thresh' in n))]


@torch.no_grad()
def measure_bn(model, label_file, data_dir=VAL_DIR, names=None):
    """Mean/var of every BN input (eval mode) over pixels, averaged over images."""
    names = names or bn_names(model)
    mods = dict(model.named_modules())
    acc = {n: [0., 0., 0] for n in names}
    def mk(n):
        def h(m, inp, out):
            x = inp[0].double(); a = acc[n]
            a[0] = a[0] + x.mean((0, 2, 3)); a[1] = a[1] + (x * x).mean((0, 2, 3)); a[2] += 1
        return h
    hs = [mods[n].register_forward_hook(mk(n)) for n in names]
    dl = loader(label_file, data_dir)
    for rec in dl:
        model(rec['image'][None].cuda())
    for h in hs: h.remove()
    out = {}
    for n in names:
        a = acc[n]; mu = a[0] / a[2]; var = a[1] / a[2] - mu ** 2
        out[n] = (mu.float().cpu(), var.clamp_min(0).float().cpu())
    return out


def apply_bn(sd, stats, names=None, alpha=1.0, what='both'):
    """Move running stats toward measured ones. var is interpolated in log space."""
    sd = {k: v.clone() for k, v in sd.items()}
    for n in (names or list(stats)):
        mu, var = stats[n]
        if what in ('both', 'mean'):
            sd[n + '.running_mean'] = ((1 - alpha) * sd[n + '.running_mean'] + alpha * mu).to(sd[n + '.running_mean'].dtype)
        if what in ('both', 'var'):
            rv = sd[n + '.running_var']
            sd[n + '.running_var'] = torch.exp((1 - alpha) * torch.log(rv + 1e-5) + alpha * torch.log(var + 1e-5)).to(rv.dtype)
    return sd


def recalibrate(sd0, label_file, names_filter=None, rounds=4, what='both', log=print):
    """Fixed-point recalibration: measure with the current model, set all selected layers, repeat."""
    sd = {k: v.clone() for k, v in sd0.items()}
    for r in range(rounds):
        model = make_model(sd)
        names = [n for n in bn_names(model) if names_filter is None or names_filter(n)]
        st = measure_bn(model, label_file, names=names)
        ch = max(float(((st[n][1] + 1e-5) / (sd[n + '.running_var'] + 1e-5)).log2().abs().mean()) for n in names)
        log(f'  round {r}: max over layers of mean|log2 var ratio| before update = {ch:.4f}')
        sd = apply_bn(sd, st, names, 1.0, what)
        del model
    return sd


def eval_sd(sd, label_file, base_cnt=None, tag='', log=print):
    model = make_model(sd)
    r = run(model, label_file)
    w = evalpy_weights(r['empty'])
    s = f"{tag:38s} evalpy {fmt(prf(r['cnt'], w))} | all {fmt(prf(r['cnt']))}"
    if base_cnt is not None:
        s += f"\n{'':38s} evalpy {fmt_boot(paired_bootstrap(base_cnt, r['cnt'], w))}\n{'':38s} all    {fmt_boot(paired_bootstrap(base_cnt, r['cnt']))}"
    log(s)
    return r
