"""Contact sheets of ALL background/empty-image FPs on val_dev for a manual tally."""
import os, sys, pickle, json
sys.path.insert(0, os.path.dirname(__file__))
from taxonomy import *
r = pickle.load(open(os.path.join(WS, 'cache_base_val_dev.pkl'), 'rb'))
tax = json.load(open(os.path.join(WS, 'taxonomy_base_dev.json')))
items = []
for t in tax:
    i = t['i']
    for x in t['dets']:
        if x['cat'] in ('fp_background', 'fp_empty_image'):
            items.append((i, r['boxes'][i][x['d']], x['cat'], x['d']))
json.dump([(i, c, d) for i, _, c, d in items], open(os.path.join(WS, 'fp_bg_index.json'), 'w'))
os.makedirs(os.path.join(WS, 'viz', 'fp_all'), exist_ok=True)
for s in range(0, len(items), 35):
    crop_sheet(r, VAL_DIR, [(i, b, f"#{s + k} {'E' if c == 'fp_empty_image' else 'B'} img{i}") for k, (i, b, c, d) in enumerate(items[s:s + 35])],
               os.path.join(WS, 'viz', 'fp_all', f'sheet_{s // 35}.jpg'), cols=7, cell=(250, 170), pad=0.5, maxn=35)
print(len(items))
