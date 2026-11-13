"""Does overlapping-crop (tiled / sliced) inference or multi-scale fusion raise recall? Original weights, forward only.
usage: 98_tiles.py <label_file> <tag> [data_dir]
Variants (probability maps fused at the pixel level, then the fixed postprocess):
  F{z}        whole frame resized with limit 736*z (F1.0 = the baseline protocol)
  T{z}/{t}    frame at zoom z cut into t x t tiles, 25% overlap, centre-weighted blend ("b") or max ("m") in overlaps
  A+B max/mean  two maps resized to the larger one and fused"""
import os, sys, pickle, itertools, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lf, tag = sys.argv[1:3]; dd = sys.argv[3] if len(sys.argv) > 3 else VAL_DIR
ZOOMS = [0.75, 1.0, 1.5, 2.0]
TILES = [(1.0, 480), (1.5, 736), (2.0, 736), (2.0, 960)]


class Multi(torch.utils.data.Dataset):
    def __init__(self):
        self.sets = {z: EvalSet(lf, dd, dict(limit_side_len=int(round(736 * z / 32)) * 32, limit_type='min')) for z in ZOOMS}
    def __len__(self): return len(self.sets[1.0])
    def __getitem__(self, i):
        out = {z: self.sets[z][i] for z in ZOOMS}
        return i, out


def starts(n, t):
    if n <= t: return [0]
    k = int(np.ceil((n - t) / (t * 0.75))) + 1
    return sorted({int(round(s / 32)) * 32 if s + t < n else n - t for s in np.linspace(0, n - t, k)})


@torch.no_grad()
def tiled(model, x, t):
    """x: 1x3xHxW normalised. returns (blend, max) prob maps HxW."""
    H, W = x.shape[2:]
    th, tw = min(t, H), min(t, W)
    ys, xs = starts(H, th), starts(W, tw)
    crops = torch.cat([x[:, :, y:y + th, c:c + tw] for y in ys for c in xs])
    pr = torch.cat([model(crops[k:k + 8])['res'][:, 0].float() for k in range(0, len(crops), 8)])
    ry = torch.minimum(torch.arange(th, device=x.device) + 1, th - torch.arange(th, device=x.device)).float()
    rx = torch.minimum(torch.arange(tw, device=x.device) + 1, tw - torch.arange(tw, device=x.device)).float()
    wt = (ry[:, None] * rx[None, :]); wt = wt / wt.max() + 1e-3
    acc = torch.zeros(H, W, device=x.device); den = torch.zeros(H, W, device=x.device); mx = torch.zeros(H, W, device=x.device)
    for k, (y, c) in enumerate(itertools.product(ys, xs)):
        acc[y:y + th, c:c + tw] += pr[k] * wt; den[y:y + th, c:c + tw] += wt
        mx[y:y + th, c:c + tw] = torch.maximum(mx[y:y + th, c:c + tw], pr[k])
    return acc / den, mx


def fuse(a, b, mode):
    if a.shape != b.shape:
        big, small = (a, b) if a.numel() >= b.numel() else (b, a)
        small = Fn.interpolate(small[None, None], size=big.shape, mode='bilinear', align_corners=False)[0, 0]
        a, b = big, small
    return torch.maximum(a, b) if mode == 'max' else (a + b) / 2


torch.set_grad_enabled(False)
model = make_model(); post = DBPostProcess(**FIXED_PP)
dl = torch.utils.data.DataLoader(Multi(), batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda b: b[0])
N = len(dl.dataset); cnt = {}; ntile = {}
items = dl.dataset.sets[1.0].items
empty = np.array([len(json.loads(l)) == 0 for _, l in items])
for i, recs in dl:
    maps = {}
    for z in ZOOMS:
        x = recs[z]['image'][None].cuda()
        maps[f'F{z}'] = (model(x)['res'][0, 0].float(), recs[z]['shape'])
        for zz, t in TILES:
            if zz == z:
                b, m = tiled(model, x, t)
                maps[f'T{z}/{t}b'] = (b, recs[z]['shape']); maps[f'T{z}/{t}m'] = (m, recs[z]['shape'])
    base = maps['F1.0']
    def big_shape(*ks): return max((maps[k] for k in ks), key=lambda v: v[0].numel())[1]
    for other in ['F0.75', 'F1.5', 'F2.0', 'T1.0/480b', 'T1.5/736b', 'T2.0/736b', 'T2.0/960b']:
        for mode in ['max', 'mean']:
            maps[f'F1.0+{other} {mode}'] = (fuse(base[0], maps[other][0], mode), big_shape('F1.0', other))
    m3 = fuse(fuse(maps['F0.75'][0], base[0], 'max'), maps['F1.5'][0], 'max'); maps['F0.75+F1.0+F1.5 max'] = (m3, maps['F1.5'][1])
    m3 = (fuse(maps['F0.75'][0], maps['F1.5'][0], 'mean') * 2 + fuse(base[0], maps['F1.5'][0], 'mean') * 2 - maps['F1.5'][0]) / 3
    maps['F0.75+F1.0+F1.5 mean'] = (m3, maps['F1.5'][1])
    r = recs[1.0]
    for k, (p, shp) in maps.items():
        boxes = postprocess(p.cpu().numpy(), shp, post)
        cnt.setdefault(k, np.zeros((N, 3), np.int64))[i] = score_image(boxes, r['polys'], r['tags'])
w = evalpy_weights(empty); b = cnt['F1.0']
print(f'=== {tag}: {N} images, {int(empty.sum())} empty; fixed pp; original weights')
print(f"{'variant':28s} | all P / R / H        | evalpy P / R     | boxes | GT +/-   | dR all [95%]              | dP all [95%]")
for k, c in cnt.items():
    d = c[:, 2] - b[:, 2]; a = prf(c); e = prf(c, w); bt = paired_bootstrap(b, c, B=4000)
    print(f"{k:28s} | {a['precision']:.4f} {a['recall']:.4f} {a['hmean']:.4f} | {e['precision']:.4f} {e['recall']:.4f} | {a['det']:5d} | +{int(d[d > 0].sum()):2d}/{int(d[d < 0].sum()):3d} | "
          f"{bt['recall'][0]:+.4f} [{bt['recall'][1]:+.4f},{bt['recall'][2]:+.4f}] | {bt['precision'][0]:+.4f} [{bt['precision'][1]:+.4f},{bt['precision'][2]:+.4f}]")
pickle.dump(dict(cnt=cnt, empty=empty), open(os.path.join(WS, f'tiles_{tag}.pkl'), 'wb'))
