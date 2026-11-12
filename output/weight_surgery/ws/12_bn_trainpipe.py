"""Why do BN running stats differ from eval-pipeline stats? Measure the same statistics on the
TRAINING pipeline (640 crops, CopyPaste, flip/rotate/scale) with the model in eval mode, and also
what train-mode batch statistics look like on those crops (batch 8, as in training per card)."""
import os, sys, pickle, numpy as np, torch, copy
sys.path.insert(0, os.path.dirname(__file__))
from bn import *
from torchocr.data import build_dataloader
from torchocr.utils.logging import get_logger
cfg = load_cfg()
cfg['Train']['dataset']['data_dir'] = VAL_DIR
cfg['Train']['dataset']['label_file_list'] = [os.path.join(VAL_DIR, 'train.txt')]
cfg['Train']['loader']['num_workers'] = 3
cfg['Train']['loader']['batch_size_per_card'] = 8
torch.manual_seed(0); np.random.seed(0); import random; random.seed(0)
dl = build_dataloader(cfg, 'Train', get_logger())
model = make_model()
names = bn_names(model); mods = dict(model.named_modules())
acc = {n: [0., 0., 0, []] for n in names}
def mk(n):
    def h(m, inp, out):
        x = inp[0].double(); a = acc[n]
        a[0] = a[0] + x.mean((0, 2, 3)); a[1] = a[1] + (x * x).mean((0, 2, 3)); a[2] += 1
        a[3].append(x.var((0, 2, 3)).float().cpu())   # per-batch variance (what train-mode BN would use)
    return h
hs = [mods[n].register_forward_hook(mk(n)) for n in names]
nb = 0
with torch.no_grad():
    for batch in dl:
        model(batch[0].cuda()); nb += 1
        if nb >= 100: break
sd = model.state_dict()
ev = pickle.load(open(os.path.join(WS, 'bn_stats_train800.pkl'), 'rb'))['stats']
print(f'{nb} batches of 8 train crops')
print(f"{'layer':52s} log2(var/running_var): eval-pipe | train-pipe pooled | train-pipe median batch | train-pipe max batch")
rows = []
for n in names:
    a = acc[n]; mu = (a[0] / a[2]).float().cpu(); var = (a[1] / a[2]).float().cpu() - mu ** 2
    rv = sd[n + '.running_var'].cpu() + 1e-5
    pb = torch.stack(a[3])
    r = [float(torch.log2((ev[n][1] + 1e-5) / rv).mean()), float(torch.log2((var + 1e-5) / rv).mean()),
         float(torch.log2((pb.median(0).values + 1e-5) / rv).mean()), float(torch.log2((pb.max(0).values + 1e-5) / rv).mean())]
    rows.append(r)
    if any(s in n for s in ['stem4', 's1.1.channel_mixer.compress', 's2.2.channel_mixer.compress', 's3.4.channel_mixer.compress', 's4.0.channel_mixer.compress', 's4.2', 'neck', 'head']):
        print(f'{n:52s} {r[0]:+.2f} | {r[1]:+.2f} | {r[2]:+.2f} | {r[3]:+.2f}')
print('mean over all layers:', np.mean(rows, 0).round(3))
pickle.dump({n: ((acc[n][0] / acc[n][2]).float().cpu(), ((acc[n][1] / acc[n][2]).float().cpu() - (acc[n][0] / acc[n][2]).float().cpu() ** 2)) for n in names},
            open(os.path.join(WS, 'bn_stats_trainpipe.pkl'), 'wb'))
