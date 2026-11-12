import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from torchocr import Trainer
cfgo = Config(CFG_PATH)
cfgo.merge_dict({'Global.pretrained_model': ORIG, 'Eval.dataset.data_dir': VAL_DIR,
                 'Eval.dataset.label_file_list': [os.path.join(VAL_DIR, 'val.txt')], 'Global.output_dir': WS + '/logs/tr'})
tr = Trainer(cfgo, mode='eval')
print(type(tr.post_process_class), vars(tr.post_process_class))
tr.model.eval()
mine = make_model()
post = DBPostProcess(**FIXED_PP)
tot = np.zeros(3); tot2 = np.zeros(3); nd = 0
sd1 = tr.model.state_dict(); sd2 = mine.state_dict()
print('sd equal', all(torch.equal(sd1[k], sd2[k]) for k in sd2), tr.model.training)
with torch.no_grad():
    for i, batch in enumerate(tr.valid_dataloader):
        batch = [t.to(dtype=torch.float32, device='cuda') for t in batch]
        p1 = tr.model(batch[0], data=batch[1:]); p2 = mine(batch[0])
        r1 = tr.post_process_class(p1, batch); r2 = post(p2, batch)
        g = (batch[2].cpu().numpy()[0], batch[3].cpu().numpy()[0])
        c1 = score_image(r1[0]['points'], *g); c2 = score_image(r2[0]['points'], *g)
        tot += c1; tot2 += c2
        if c1 != c2 and nd < 6:
            nd += 1; print(i, c1, c2, float((p1['res'] - p2['res']).abs().max()))
print(tot, tot2)
