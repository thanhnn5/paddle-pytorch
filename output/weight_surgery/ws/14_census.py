"""H5: dead / saturated / redundant units. Activation census on 240 train images (eval pipeline):
 - BN channels: stored running_var ~ 0 ("dead in training") vs variance actually seen at eval,
   and the eval-time output variance of those channels (noise amplification through eps=1e-5).
 - SE gates (5 backbone, 8 neck): mean gate, share of (image, channel) saturated at 0 or 1, std over images.
 - RepLK neck branches: share of output energy per branch."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from bn import *
SUB = os.path.join(WS, 'train_sub800.txt')
ds = EvalSet(SUB, VAL_DIR)
imgs = [ds[i]['image'][None].cuda() for i in range(0, 800, 4)][:200]
model = make_model(); mods = dict(model.named_modules())
bn_in, bn_out, se, br = {}, {}, {}, {}
hs = []
for n in bn_names(model):
    def h(m, i, o, n=n):
        x = i[0].double(); a = bn_in.setdefault(n, [0., 0.]); a[0] = a[0] + x.mean((0, 2, 3)); a[1] = a[1] + (x * x).mean((0, 2, 3))
        y = o.double(); b = bn_out.setdefault(n, [0., 0.]); b[0] = b[0] + y.mean((0, 2, 3)); b[1] = b[1] + (y * y).mean((0, 2, 3))
    hs.append(mods[n].register_forward_hook(h))
from torchocr.modeling.backbones.det_pplcnetv4 import SELayer
from torchocr.modeling.necks.db_fpn import PaddleSEModule, DilatedReparamBlock
for n, m in model.named_modules():
    if isinstance(m, SELayer):
        hs.append(m.hardsigmoid.register_forward_hook(lambda mm, i, o, n=n: se.setdefault(n, []).append(o.flatten().cpu())))
    if isinstance(m, PaddleSEModule):
        def h(mm, i, o, n=n):
            x = i[0].mean(dim=(2, 3), keepdim=True); g = torch.clamp(0.2 * mm.conv2(torch.relu(mm.conv1(x))) + 0.5, 0, 1)
            se.setdefault(n, []).append(g.flatten().cpu())
        hs.append(m.register_forward_hook(h))
    if isinstance(m, DilatedReparamBlock):
        for bn in ['origin_bn'] + [f'dil_bn_k{k}_{r}' for k, r in zip(m.kernel_sizes, m.dilates)]:
            hs.append(getattr(m, bn).register_forward_hook(lambda mm, i, o, key=(n, bn): br.__setitem__(key, br.get(key, 0.) + float((o.double() ** 2).mean()))))
with torch.no_grad():
    for x in imgs: model(x)
N = len(imgs); sd = model.state_dict()
print('--- BN channels with stored running_var < 1e-6 (never active in training)')
dead = {}
for n in bn_in:
    rv = sd[n + '.running_var'].cpu(); d = rv < 1e-6
    if d.any():
        mu = (bn_in[n][0] / N).cpu(); var = (bn_in[n][1] / N).cpu() - mu ** 2
        om = (bn_out[n][0] / N).cpu(); ov = (bn_out[n][1] / N).cpu() - om ** 2
        g = sd[n + '.weight'].cpu().abs()
        dead[n] = d
        print(f"{n:50s} dead {int(d.sum()):3d}/{d.numel()}  eval input var on dead: max {float(var[d].max()):.2e}  |gamma| on dead: mean {float(g[d].mean()):.3f}"
              f"  eval OUTPUT var: dead mean {float(ov[d].mean()):.3e} max {float(ov[d].max()):.3e} | live mean {float(ov[~d].mean()):.3e}")
pickle.dump(dead, open(os.path.join(WS, 'dead_bn_channels.pkl'), 'wb'))
print('--- SE gates')
for n, v in se.items():
    g = torch.stack(v)  # N x C
    print(f"{n:42s} C={g.shape[1]:3d} mean {float(g.mean()):.3f}  sat@0 {float((g <= 0).float().mean()):.3f}  sat@1 {float((g >= 1).float().mean()):.3f}"
          f"  channels always 0: {int((g <= 0).all(0).sum())}  always 1: {int((g >= 1).all(0).sum())}  mean std over images {float(g.std(0).mean()):.4f}")
print('--- RepLK neck: share of branch output energy (E[y^2] after BN)')
for i in range(4):
    ks = [k for k in br if k[0] == f'neck.inp_conv_dw.{i}']; tot = sum(br[k] for k in ks)
    print(f'neck.inp_conv_dw.{i}', ' '.join(f"{k[1]}={br[k] / tot:.3f}" for k in ks))
