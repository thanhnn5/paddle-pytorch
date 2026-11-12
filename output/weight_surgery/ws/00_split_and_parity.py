"""Write val_dev/val_holdout label files; check harness parity with tools/eval.py on full val."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *

src = os.path.join(VAL_DIR, 'val.txt')
lines = open(src, 'rb').read().decode('utf-8').split('\n')
lines = [l for l in lines if l]
parts = {'dev': [], 'holdout': []}
for l in lines:
    parts[split_of(l.split('\t')[0])].append(l)
for k, v in parts.items():
    with open(os.path.join(WS, f'val_{k}.txt'), 'w') as f:
        f.write('\n'.join(v) + '\n')
    print(k, len(v))

model = make_model()
res = run(model, src)
w = evalpy_weights(res['empty'])
print('full val, evalpy protocol:', fmt(prf(res['cnt'], w)))
print('full val, all images     :', fmt(prf(res['cnt'])))
ne = ~res['empty']
print('full val, non-empty only :', fmt(prf(res['cnt'][ne])))
print('empty images:', res['empty'].sum(), 'dets on empty images:', res['cnt'][res['empty'], 1].sum(),
      'empty images with >=1 det:', (res['cnt'][res['empty'], 1] > 0).sum())
np.save(os.path.join(WS, 'cache_base_fullval_cnt.npy'), res['cnt'])
