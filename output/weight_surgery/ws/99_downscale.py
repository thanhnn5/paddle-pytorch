"""Multi-scale fusion with smaller scales (follow-up of 98_tiles.py). usage: 99_downscale.py <label_file> <tag> [data_dir]"""
import os, sys, pickle, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lf, tag = sys.argv[1:3]; dd = sys.argv[3] if len(sys.argv) > 3 else VAL_DIR
ZOOMS = [0.5, 0.625, 0.75, 0.875, 0.9375, 1.0, 1.0625, 1.125]
torch.set_grad_enabled(False)
class Multi(torch.utils.data.Dataset):
    """z = 1.0 is the config's pipeline. Other zooms: own resize to 736*z on the short side (DetResizeForTest with
    limit_type 'min' never downsizes), multiples of 32, then the config's normalisation."""
    def __init__(self): self.base = EvalSet(lf, dd); self.items = self.base.items
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        out = {1.0: self.base[i]}
        with open(os.path.join(dd, self.items[i][0]), 'rb') as f: img = transform({'image': f.read(), 'label': ''}, self.base.decode)['image']
        h, w = img.shape[:2]
        for z in ZOOMS:
            if z == 1.0: continue
            r = 736 * z / min(h, w); rh, rw = max(32, int(round(h * r / 32)) * 32), max(32, int(round(w * r / 32)) * 32)
            d = {'image': cv2.resize(img, (rw, rh))}
            for op in self.base.post[1:]: d = op(d)
            out[z] = dict(image=torch.from_numpy(np.ascontiguousarray(d['image'])))
        return i, out
def up(a, size): return a if a.shape == size else Fn.interpolate(a[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]
model = make_model(); post = DBPostProcess(**FIXED_PP)
dl = torch.utils.data.DataLoader(Multi(), batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda b: b[0])
N = len(dl.dataset); cnt = {}
empty = np.array([len(json.loads(l)) == 0 for _, l in dl.dataset.items])
for i, recs in dl:
    P = {z: model(recs[z]['image'][None].cuda())['res'][0, 0].float() for z in ZOOMS}
    S = P[1.0].shape; shp = recs[1.0]['shape']; U = {z: up(P[z], S) for z in ZOOMS}
    maps = {f'F{z}': U[z] for z in ZOOMS}
    for z in ZOOMS:
        if z != 1.0:
            maps[f'F1.0+F{z} max'] = torch.maximum(U[1.0], U[z]); maps[f'F1.0+F{z} mean'] = (U[1.0] + U[z]) / 2
    for nm, zs in [('0.9375,1,1.0625', [0.9375, 1.0, 1.0625]), ('0.875,1,1.125', [0.875, 1.0, 1.125]), ('0.75,1', [0.75, 1.0]), ('0.75,0.875,1', [0.75, 0.875, 1.0]), ('0.5,0.75,1', [0.5, 0.75, 1.0]), ('0.875..1.125 x5', [0.875, 0.9375, 1.0, 1.0625, 1.125])]:
        st = torch.stack([U[z] for z in zs]); maps[f'max {nm}'] = st.max(0)[0]; maps[f'mean {nm}'] = st.mean(0)
    for k, p in maps.items():
        cnt.setdefault(k, np.zeros((N, 3), np.int64))[i] = score_image(postprocess(p.cpu().numpy(), shp, post), recs[1.0]['polys'], recs[1.0]['tags'])
w = evalpy_weights(empty); b = cnt['F1.0']
print(f'=== {tag}: {N} images, {int(empty.sum())} empty; fixed pp; original weights; all maps fused at the F1.0 resolution')
for k, c in cnt.items():
    d = c[:, 2] - b[:, 2]; a = prf(c); e = prf(c, w); bt = paired_bootstrap(b, c, B=4000)
    print(f"{k:34s} | all {a['precision']:.4f} {a['recall']:.4f} {a['hmean']:.4f} | evalpy {e['precision']:.4f} {e['recall']:.4f} | boxes {a['det']:5d} | +{int(d[d > 0].sum()):2d}/{int(d[d < 0].sum()):3d} | "
          f"dR {bt['recall'][0]:+.4f} [{bt['recall'][1]:+.4f},{bt['recall'][2]:+.4f}] | dP {bt['precision'][0]:+.4f} [{bt['precision'][1]:+.4f},{bt['precision'][2]:+.4f}]")
pickle.dump(dict(cnt=cnt, empty=empty), open(os.path.join(WS, f'downscale_{tag}.pkl'), 'wb'))
