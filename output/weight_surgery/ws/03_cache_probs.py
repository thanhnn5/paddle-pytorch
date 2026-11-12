"""Cache probability maps (float16) + GT + shapes for a label file and checkpoint.
usage: 03_cache_probs.py <label_file> <ckpt> <out.pkl> [data_dir]"""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lf, ck, out = sys.argv[1:4]
dd = sys.argv[4] if len(sys.argv) > 4 else VAL_DIR
model = make_model(load_sd(ck))
r = run(model, lf, dd, keep_prob=True, keep_boxes=True)
print(os.path.basename(lf), 'evalpy:', fmt(prf(r['cnt'], evalpy_weights(r['empty']))))
print(os.path.basename(lf), 'all   :', fmt(prf(r['cnt'])))
r['probs'] = [p.astype(np.float16) for p in r['probs']]
pickle.dump(r, open(out, 'wb'), protocol=4)
