"""Shared tooling for the weight-surgery experiment.

Reuses the repo's model builder, eval transforms, DBPostProcess and
DetectionIoUEvaluator so numbers agree with tools/eval.py.

Two aggregation protocols are reported:
  * "all"    : every line of the label file is one image (empty-label images
               count: their predictions are false positives).
  * "evalpy" : what tools/eval.py really computes. SimpleDataSet drops a sample
               whose label list is empty (DetLabelEncode returns None) and
               substitutes sample (idx+1) % N, so empty images are never scored
               and the image after a run of empties is counted several times.
"""
import os, sys, json, copy, hashlib
import numpy as np
import torch
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
from torchocr import Config
from torchocr.modeling.architectures import build_model
from torchocr.data.imaug import create_operators, transform
from torchocr.postprocess.db_postprocess import DBPostProcess
from torchocr.metrics.eval_det_iou import DetectionIoUEvaluator

CFG_PATH = os.path.join(ROOT, 'configs/det/PP-OCRv6/PP-OCRv6_small_det.yml')
ORIG = os.path.join(ROOT, 'weights/best_accuracy.pth')
WS = os.path.join(ROOT, 'output/weight_surgery')
VAL_DIR = os.path.join(ROOT, 'data/ocr_det_mask_3-1')
FIXED_PP = dict(thresh=0.2, box_thresh=0.45, unclip_ratio=1.4, max_candidates=3000)


def load_cfg():
    return Config(CFG_PATH).cfg


def load_sd(path=ORIG):
    return torch.load(path, map_location='cpu')['state_dict']


def save_sd(sd, path, ref=None):
    """Save an edited state_dict, checking names/shapes/dtypes against the original."""
    ref = ref if ref is not None else load_sd()
    assert set(sd.keys()) == set(ref.keys()), 'tensor names changed'
    sd = {k: sd[k] for k in ref}  # keep the original key order
    for k in ref:
        assert sd[k].shape == ref[k].shape and sd[k].dtype == ref[k].dtype, k
        assert torch.isfinite(sd[k].float()).all(), k
    torch.save({'state_dict': {k: v.cpu() for k, v in sd.items()}}, path)


def make_model(sd=None, device='cuda'):
    cfg = load_cfg()
    model = build_model(copy.deepcopy(cfg['Architecture']))
    if sd is None:
        sd = load_sd()
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def read_label_file(path):
    items = []
    with open(path, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8').strip('\n')
            name, label = line.split('\t')
            items.append((name, label))
    return items


class EvalSet(torch.utils.data.Dataset):
    """Eval pipeline of the config, but keeps images with an empty label list."""

    def __init__(self, label_file, data_dir, resize_kwargs=None):
        self.items = read_label_file(label_file)
        self.data_dir = data_dir
        cfg = load_cfg()
        tfs = copy.deepcopy(cfg['Eval']['dataset']['transforms'])
        names = [list(t)[0] for t in tfs]
        assert names == ['DecodeImage', 'DetLabelEncode', 'DetResizeForTest',
                         'NormalizeImage', 'ToCHWImage', 'KeepKeys'], names
        if resize_kwargs:
            tfs[2]['DetResizeForTest'] = dict(resize_kwargs)
        self.decode = create_operators(tfs[:1])
        self.encode = create_operators(tfs[1:2])[0]
        self.post = create_operators(tfs[2:5])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, label = self.items[i]
        with open(os.path.join(self.data_dir, name), 'rb') as f:
            data = {'image': f.read(), 'label': label}
        data = transform(data, self.decode)
        if len(json.loads(label)) == 0:
            polys = np.zeros((0, 4, 2), np.float32)
            tags = np.zeros((0,), np.bool_)
        else:
            d = self.encode({'label': label})
            polys, tags = d['polys'], d['ignore_tags']
        data = transform(data, self.post)
        return dict(idx=i, image=torch.from_numpy(np.ascontiguousarray(data['image'])),
                    shape=data['shape'].astype(np.float32), polys=polys, tags=tags)


def loader(label_file, data_dir, resize_kwargs=None, workers=3):
    ds = EvalSet(label_file, data_dir, resize_kwargs)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                                       collate_fn=lambda b: b[0], prefetch_factor=4)


_EV = DetectionIoUEvaluator()


def score_image(boxes, polys, tags):
    gt = [{'points': p, 'text': '', 'ignore': t} for p, t in zip(polys, tags)]
    det = [{'points': b, 'text': ''} for b in boxes]
    r = _EV.evaluate_image(gt, det)
    return r['gtCare'], r['detCare'], r['detMatched']


def postprocess(prob, shape, pp):
    """prob: HxW float32 probability map; shape: [src_h, src_w, rh, rw]."""
    post = pp if isinstance(pp, DBPostProcess) else DBPostProcess(**pp)
    out = post({'res': prob[None, None]}, [None, np.asarray(shape)[None]])
    return out[0]['points']


@torch.no_grad()
def run(model, label_file, data_dir=VAL_DIR, pp=FIXED_PP, resize_kwargs=None, keep_prob=False,
        keep_boxes=False, hook=None, desc=None, amp=False):
    """Run model over a label file. Returns dict with per-image counts array
    cnt[N,3] = (gtCare, detCare, detMatched), names, and optionally probs/boxes.
    `hook(i, rec)` is called per image after the forward pass if given."""
    post = DBPostProcess(**pp)
    dl = loader(label_file, data_dir, resize_kwargs)
    N = len(dl.dataset)
    cnt = np.zeros((N, 3), np.int64)
    probs, boxes_all, shapes, gts = [None] * N, [None] * N, [None] * N, [None] * N
    for rec in dl:
        i = rec['idx']
        x = rec['image'][None].cuda(non_blocking=True)
        prob = model(x)['res'][0, 0].float().cpu().numpy()
        boxes = postprocess(prob, rec['shape'], post)
        cnt[i] = score_image(boxes, rec['polys'], rec['tags'])
        shapes[i] = rec['shape']
        gts[i] = (rec['polys'], rec['tags'])
        if keep_prob:
            probs[i] = prob
        if keep_boxes:
            boxes_all[i] = boxes
        if hook is not None:
            hook(i, rec)
    names = [n for n, _ in dl.dataset.items]
    empty = np.array([len(json.loads(l)) == 0 for _, l in dl.dataset.items])
    return dict(cnt=cnt, names=names, empty=empty, probs=probs, boxes=boxes_all, shapes=shapes, gts=gts)


def evalpy_weights(empty):
    """Multiplicity of each image under tools/eval.py's SimpleDataSet substitution."""
    N = len(empty)
    w = np.zeros(N, np.int64)
    for i in range(N):
        j = i
        while empty[j]:
            j = (j + 1) % N
        w[j] += 1
    return w


def prf(cnt, w=None):
    if w is None:
        w = np.ones(len(cnt))
    g, d, m = (cnt * np.asarray(w)[:, None]).sum(0)
    r = m / g if g else 0.
    p = m / d if d else 0.
    h = 2 * p * r / (p + r) if p + r else 0.
    return dict(precision=p, recall=r, hmean=h, gt=int(g), det=int(d), tp=int(m))


def fmt(m):
    return f"P {m['precision']:.4f} R {m['recall']:.4f} H {m['hmean']:.4f} (gt {m['gt']} det {m['det']} tp {m['tp']})"


def paired_bootstrap(cnt_a, cnt_b, w=None, B=10000, seed=0):
    """Paired bootstrap over images of (b - a) for precision/recall/hmean.
    Returns dict metric -> (delta, lo, hi, p_le0)."""
    rng = np.random.default_rng(seed)
    N = len(cnt_a)
    w = np.ones(N) if w is None else np.asarray(w, float)
    a = cnt_a * w[:, None]
    b = cnt_b * w[:, None]
    idx = rng.integers(0, N, size=(B, N))
    def agg(c):
        s = c[idx].sum(1).astype(float)  # B x 3
        g, d, m = s[:, 0], s[:, 1], s[:, 2]
        r = np.divide(m, g, out=np.zeros(B), where=g > 0)
        p = np.divide(m, d, out=np.zeros(B), where=d > 0)
        h = np.divide(2 * p * r, p + r, out=np.zeros(B), where=(p + r) > 0)
        return p, r, h
    pa, pb = agg(a), agg(b)
    ma, mb = prf(cnt_a, w), prf(cnt_b, w)
    out = {}
    for k, xa, xb in zip(['precision', 'recall', 'hmean'], pa, pb):
        d = xb - xa
        out[k] = (mb[k] - ma[k], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean()))
    return out


def fmt_boot(bt):
    return ' | '.join(f"d{k[0].upper()} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]" for k, v in bt.items())


def split_of(name):
    """Deterministic split of val by image name: md5(basename) parity."""
    h = hashlib.md5(os.path.basename(name).encode()).hexdigest()
    return 'dev' if int(h, 16) % 2 == 0 else 'holdout'
