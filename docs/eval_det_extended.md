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

## Results — teacher vs student (both @ 1280-max)

Teacher checkpoint: `output/PP-OCRv5_convnext_det_unfreeze/best.pth` (the
DINOv3-ConvNeXt-tiny detection model used as the Stage-2 distill source and
the AuxDistill target during femto fine-tune).

| metric                       | femto sal-mix-aux | **convnext-tiny (teacher)** | Δ (teacher − student) |
|------------------------------|-------------------|------------------------------|------|
| Total predictions            | 1689              | 1343                         | -346 |
| **P @ IoU 0.3**              | 0.677             | **0.844**                    | +16.7 |
| R @ IoU 0.3                  | 0.902             | 0.895                        | -0.7  |
| **F1 @ IoU 0.3**             | 0.773             | **0.869**                    | +9.6  |
| P @ IoU 0.5                  | 0.650             | **0.822**                    | +17.2 |
| F1 @ IoU 0.5                 | 0.743             | **0.846**                    | +10.3 |
| F1 @ IoU 0.7                 | 0.482             | **0.600**                    | +11.8 |
| mean IoU of TPs              | 0.732             | 0.746                        | +1.4  |
| Small-text recall (IoU 0.5)  | 0.765             | 0.756                        | -0.9  |
| Medium recall                | 0.908             | 0.934                        | +2.6  |
| Large recall                 | 0.927             | 0.924                        | -0.3  |
| Images w/ 0 FPs              | 82.7 %            | **93.2 %**                   | +10.5 |
| Mean #pred − #gt drift       | +0.14             | +0.03                        | -0.11 |

**Read:**
- **Teacher's precision is dramatically higher** (+17 P, +10 F1). Fewer FPs
  per image (93 % clean vs 83 %).
- **Recall is essentially tied** — the student isn't missing more boxes, it
  just adds spurious extras. AuxDistill@5.0 narrowed but didn't close the gap.
- **Small-text recall (~0.76) is the bottleneck for both** — a resolution /
  augmentation problem, not a capacity problem.
- Mean IoU of TPs is only +1.4 in teacher's favour — localization tightness
  is comparable.
- **Distillation headroom: ~+10 F1 at IoU 0.5 still on the table.**
  Stage-2 feature alignment is already strong (probe mse_ln s3 ≈ 0.131); the
  next lever is making head/neck convert that into precision parity — try
  raising AuxDistill weight or co-distilling head logits.
