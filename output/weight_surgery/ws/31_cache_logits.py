"""Cache pre-sigmoid logits (float16) + GT + shapes. usage: 31_cache_logits.py <label_file> <ckpt> <out.pkl> [data_dir]"""
import os, sys, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lf, ck, out = sys.argv[1:4]; dd = sys.argv[4] if len(sys.argv) > 4 else VAL_DIR
model = make_model(load_sd(ck)); L = {}
model.head.binarize.conv3.register_forward_hook(lambda m, i, o: L.__setitem__('l', o[0, 0].float().cpu().numpy().astype(np.float16)))
logits = {}
r = run(model, lf, dd, hook=lambda i, rec: logits.__setitem__(i, L['l']))
r['logits'] = [logits[i] for i in range(len(r['names']))]; r['probs'] = None
print(os.path.basename(lf), 'evalpy:', fmt(prf(r['cnt'], evalpy_weights(r['empty']))), '| all:', fmt(prf(r['cnt'])))
pickle.dump(r, open(out, 'wb'), protocol=4)
