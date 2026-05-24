# Extended detection evaluation

`tools/eval_det_extended.py` reports detection metrics that complement the
single-IoU P/R/F1 in `visualize_detections.ipynb`:

1. P/R/F1 swept over IoU thresholds {0.3, 0.5, 0.7, 0.9} + mean IoU of TPs
2. TP-IoU distribution (percentiles) — localization-quality signal
3. Size-stratified recall (S/M/L tertiles by GT polygon area)
4. Per-image FP/FN count histograms
5. Prediction-count drift (#pred − #gt per image)

Usage:

```bash
python tools/eval_det_extended.py \
    --gt   data/ocr_test_20251127/test.txt \
    --pred output/predictions/<run>/predict_det.txt
```

## Config-merge bugfix (related)

The CLI `-o name1.name2[i].name3=val` syntax was silently broken: dotted-path
overrides could not index into list elements, so an override like

```
-o Eval.dataset.transforms[2].DetResizeForTest.limit_side_len=1280
```

created a *new sibling key* (`transforms[2]`, the literal string with brackets)
instead of modifying `transforms[2]`. The detection inference / eval would
fall back to `DetResizeForTest` defaults (`limit_side_len=736, limit_type='min'`).
Fixed in `torchocr/engine/config.py` — `_merge_dict` now resolves `name[i]`
segments against list children before assignment.

To verify on your branch:

```python
from torchocr import Config
cfg = Config('configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_balanced_det.yml')
cfg.merge_dict({'Eval.dataset.transforms[2].DetResizeForTest.limit_side_len': 1280})
assert cfg.cfg['Eval']['dataset']['transforms'][2] == \
       {'DetResizeForTest': {'limit_side_len': 1280}}
```

## Results — sal-mix + balanced + AuxDistill (femto)

Test set: `data/ocr_test_20251127` (2990 images, 1267 GT polygons).
Checkpoint: `output/PP-OCRv5_convnextv2_femto_sal_mix_balanced_aux_det/phase2_joint/best.pth`.

### Effect of the inference resize side (impact of the bugfix)

|                              | 736-min (broken)  | **1280-max (fixed)** | Δ |
|------------------------------|-------------------|----------------------|------|
| Total predictions            | 1756              | 1689                 | -67  |
| P @ IoU 0.3                  | 0.650             | **0.677**            | +2.7 |
| R @ IoU 0.3                  | 0.901             | 0.902                | +0.1 |
| F1 @ IoU 0.3                 | 0.755             | **0.773**            | +1.8 |
| P @ IoU 0.5                  | 0.626             | **0.650**            | +2.4 |
| F1 @ IoU 0.5                 | 0.728             | **0.743**            | +1.5 |
| mean IoU of TPs              | 0.733             | 0.732                | ~0   |
| Small-text recall (IoU 0.5)  | 0.758             | 0.765                | +0.7 |
| Medium recall                | 0.924             | 0.908                | -1.6 |
| Large recall                 | 0.922             | 0.927                | +0.5 |
| Images w/ 0 FPs              | 81.3 %            | **82.7 %**           | +1.4 |

**Read:** 1280-max gives ~+2 P / +2 F1 mostly via fewer FPs, not new recall.
Mean IoU of matches is unchanged, so the bigger input lifts the
discrimination boundary, not localization tightness. Small-text recall stays
the bottleneck — needs an even larger infer side or stronger small-text aug.
