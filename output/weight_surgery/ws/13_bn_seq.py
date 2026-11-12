"""H1, done properly: SEQUENTIAL BN recalibration. Layers are visited in execution order; each
layer's input statistics are measured (eval mode, train images, eval pipeline) after all layers
before it have been updated, then its running stats are set. Forward-only.
Also records the LOCAL mismatch per layer (measured vs stored, with upstream already recalibrated)."""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from bn import *
NCAL = int(sys.argv[1]) if len(sys.argv) > 1 else 240
SUB = os.path.join(WS, 'train_sub800.txt'); DEV = os.path.join(WS, 'val_dev.txt')
ds = EvalSet(SUB, VAL_DIR)
imgs = [ds[i]['image'][None].cuda() for i in range(0, 800, 800 // NCAL)][:NCAL]
print('calibration images:', len(imgs))
sd0 = load_sd(); model = make_model(sd0)
# execution order
order = []
mods = dict(model.named_modules())
hs = [mods[n].register_forward_hook(lambda m, i, o, n=n: order.append(n)) for n in bn_names(model)]
with torch.no_grad(): model(imgs[0])
for h in hs: h.remove()

class Stop(Exception): pass
local = {}
with torch.no_grad():
    for n in order:
        m = mods[n]; acc = [0., 0.]
        def h(mod, inp, out):
            x = inp[0].double(); acc[0] = acc[0] + x.mean((0, 2, 3)); acc[1] = acc[1] + (x * x).mean((0, 2, 3))
            if STOP: raise Stop()
        # the DilatedReparam branches share an input; stopping at a BN is safe only for a pure chain,
        # so only stop early inside the backbone stem/blocks is not assumed: never stop early.
        STOP = False
        hk = m.register_forward_hook(h)
        for x in imgs:
            model(x)
        hk.remove()
        mu = (acc[0] / len(imgs)); var = (acc[1] / len(imgs) - mu ** 2).clamp_min(0)
        z = ((mu.float() - m.running_mean) / (m.running_var + 1e-5).sqrt())
        lr = torch.log2((var.float() + 1e-5) / (m.running_var + 1e-5))
        local[n] = (float(z.abs().mean()), float(lr.mean()), float(lr.abs().max()), float(lr.min()), float(lr.max()))
        m.running_mean.copy_(mu.float()); m.running_var.copy_(var.float())
print(f"{'layer':52s} LOCAL mismatch with upstream recalibrated: mean|z|, mean log2 var ratio, min, max")
for n in order:
    l = local[n]; print(f'{n:52s} {l[0]:.3f} {l[1]:+.3f} {l[3]:+.2f} {l[4]:+.2f}')
sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
save_sd(sd, os.path.join(WS, f'cand_bn_seqrecal_all_train{NCAL}.pth'))
pickle.dump(dict(order=order, local=local), open(os.path.join(WS, 'bn_seq_local.pkl'), 'wb'))
base = eval_sd(sd0, DEV, tag='original')
eval_sd(sd, DEV, base['cnt'], f'sequential recal, all layers ({NCAL} img)')
# dose-response between original and recalibrated stats (mean linear, var in log space)
for a in [0.25, 0.5, 0.75]:
    sda = {k: v.clone() for k, v in sd0.items()}
    for n in order:
        sda[n + '.running_mean'] = (1 - a) * sd0[n + '.running_mean'] + a * sd[n + '.running_mean']
        sda[n + '.running_var'] = torch.exp((1 - a) * torch.log(sd0[n + '.running_var'] + 1e-5) + a * torch.log(sd[n + '.running_var'] + 1e-5))
    eval_sd(sda, DEV, base['cnt'], f'interp original->recal alpha={a}')
