"""Is the 2-detection gap to tools/eval.py explained by cudnn.benchmark=True (set by Trainer)?"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
src = os.path.join(VAL_DIR, 'val.txt')
model = make_model()
a = run(model, src, keep_prob=False)
torch.backends.cudnn.benchmark = True
diffs = []
ds = EvalSet(src, VAL_DIR)
b = run(model, src)
w = evalpy_weights(a['empty'])
print('benchmark=False:', fmt(prf(a['cnt'], w)))
print('benchmark=True :', fmt(prf(b['cnt'], w)))
ch = np.where((a['cnt'] != b['cnt']).any(1))[0]
print('images changed:', len(ch), [(int(i), a['cnt'][i].tolist(), b['cnt'][i].tolist()) for i in ch[:10]])
# size of the numeric difference on changed images
with torch.no_grad():
    for i in ch[:5]:
        x = ds[int(i)]['image'][None].cuda()
        torch.backends.cudnn.benchmark = True; p1 = model(x)['res']
        torch.backends.cudnn.benchmark = False; p0 = model(x)['res']
        xd = x.double(); md = make_model().double(); pd = md(xd)['res']
        print(int(i), 'max|bench-nobench|', float((p1 - p0).abs().max()), 'max|f32-f64|', float((p0.double() - pd).abs().max()), float((p1.double() - pd).abs().max()))
