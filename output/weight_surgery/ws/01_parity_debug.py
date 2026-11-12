import os, sys, numpy as np, torch, copy
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from torchocr.data import build_dataloader
from torchocr.utils.logging import get_logger
cfg = load_cfg()
cfg['Eval']['dataset']['data_dir'] = VAL_DIR
cfg['Eval']['dataset']['label_file_list'] = [os.path.join(VAL_DIR, 'val.txt')]
dl = build_dataloader(cfg, 'Eval', get_logger())
model = make_model()
post = DBPostProcess(**FIXED_PP)
mine = EvalSet(os.path.join(VAL_DIR, 'val.txt'), VAL_DIR)
empty = np.array([len(json.loads(l)) == 0 for _, l in mine.items]); N = len(empty)
tot = np.zeros(3); tot2 = np.zeros(3); nd = 0
with torch.no_grad():
    for i, batch in enumerate(dl):
        batch = [t.to(dtype=torch.float32, device='cuda') for t in batch]
        preds = model(batch[0])
        pr = post(preds, batch)
        c1 = score_image(pr[0]['points'], batch[2].cpu().numpy()[0], batch[3].cpu().numpy()[0])
        j = i
        while empty[j]: j = (j + 1) % N
        rec = mine[j]
        p2 = model(rec['image'][None].cuda())['res'][0, 0].cpu().numpy()
        b2 = postprocess(p2, rec['shape'], post)
        c2 = score_image(b2, rec['polys'], rec['tags'])
        tot += c1; tot2 += c2
        if c1 != c2 and nd < 8:
            nd += 1
            print(i, j, c1, c2, 'img equal', torch.equal(batch[0][0].cpu(), rec['image']),
                  'prob maxdiff', float(np.abs(preds['res'][0,0].cpu().numpy() - p2).max()),
                  batch[1].cpu().numpy(), rec['shape'], batch[2].shape, rec['polys'].shape, batch[3].cpu().numpy(), rec['tags'])
print(tot, tot2)
