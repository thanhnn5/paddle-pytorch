# Weight surgery on PP-OCRv6 small det — running log

Task prompt: `prompts/det_weight_surgery.txt`. Scripts: `output/weight_surgery/ws/`.
All numbers are precision / recall / hmean, DetMetric IoU 0.5, fixed postprocess
(thresh 0.2, box_thresh 0.45, unclip 1.4) unless stated.

## 0. Setup, reproduction, measurement protocol (2026-11-12)

- `tools/eval.py` on the unedited checkpoint, full val: **P 0.8499 R 0.9434 H 0.8943** (reproduced).
- Harness `ws/common.py` (repo model builder, eval transforms, DBPostProcess, DetectionIoUEvaluator):
  full val P 0.8490 R 0.9434 H 0.8937 (gt 1609, det 1788, tp 1518). tools/eval.py has det 1786.
  Cause: GPU numerics that differ between processes (cuDNN algorithm choice). Running Trainer's own
  model and my model side by side in one process gives 1786 for both; in a plain process both the
  repo dataloader and my loader give 1788. So the harness is at parity and **±2 detections
  (about ±0.0006 hmean on full val) is the numerical noise floor** of the measurement.
- **Measurement quirk found: tools/eval.py never scores images without labels.** `DetLabelEncode`
  returns None for an empty label list and `SimpleDataSet.__getitem__` then substitutes sample
  `(idx+1) % N`. val.txt has 184 empty-label images out of 1,553, so the "1,553 images" are 1,369
  distinct images, with the image after each empty one counted twice (gt 1609 instead of 1422).
  The model puts 171 boxes on those 184 empty images (127 of them get at least one box); none of
  that is visible in the fixed measurement. The same loader was used for training (PaddleOCR's
  SimpleDataSet behaves the same), so **the 727 empty-label images in train.txt were never trained on.**
  Consequence for the test set: 1,837 of 2,990 test images are empty, and tools/eval.py would skip
  all of them. I therefore report two aggregations everywhere:
  - `evalpy`: what tools/eval.py computes (empty images substituted) — the fixed measurement;
  - `all`: every line of the label file scored, boxes on empty images are false positives.
  Full val, `all`: P 0.7614 R 0.9402 H 0.8414 (gt 1422, det 1756, tp 1337).
- Split: md5(basename) parity -> `val_dev.txt` (784 images, 85 empty) and `val_holdout.txt` (769).
  val_dev baseline: evalpy P 0.8355 R 0.9484 H 0.8884 | all P 0.7571 R 0.9464 H 0.8413.
- Holdout evaluations used so far: **0 / 5**.

## 1. Baseline A — postprocess sweep on val_dev (`ws/04_sweepA.py`, log `logs/sweepA_dev.txt`)

Grid thresh {0.1..0.5} x box_thresh {0.3..0.8} x unclip {1.0..2.0}, 560 settings, baseline maps.
- Best evalpy hmean 0.9044 at (0.10, 0.65, 1.3); best `all` hmean 0.8672 at (0.15, 0.70, 1.5).
- The gain is almost entirely box_thresh: best-over-others hmean by box_thresh
  0.45 -> 0.8926, 0.55 -> 0.8962, 0.65 -> 0.9044, 0.70 -> 0.9033, 0.75 -> 0.8951.
  unclip is flat between 1.3 and 1.5 (0.9032-0.9044); thresh is within 0.005 everywhere.
- So the postprocess ceiling on val_dev is about **+0.015 hmean (evalpy), +0.025 (all)**, selected
  on val_dev (optimistic by construction).

## 2. Error taxonomy on val_dev (`ws/taxonomy.py`, sheets in `viz/`)

Fixed postprocess, all 784 images: 910 detections, 689 matched, 221 unmatched; 728 GT, 39 unmatched
(+3 unmatched that are counted in two categories below: 42 records).

Unmatched detections (221):
| category | n |
|---|---|
| background FP on an image that has GT (no overlap with any GT) | 102 |
| FP on an empty-label image (invisible to tools/eval.py) | 84 |
| split (several boxes inside one GT) | 16 |
| tight (IoU 0.3-0.5, box inside GT) | 7 |
| fragment inside GT, IoU < 0.3 | 4 |
| loose / very loose | 6 |
| low overlap | 2 |
| merge | 0 |

Unmatched GT (42): missed entirely 18, IoU 0.3-0.5 geometry 10, partial < 0.3 9, split 5.
Matched boxes: mean IoU 0.851, only 2.5% in (0.5, 0.6] — geometry is not the problem.

**84% of the unmatched detections (186/221) are boxes on something the annotation does not mark.**
I looked at all 186 crops (`viz/fp_all/sheet_*.jpg`) and tallied by eye:
| what the box is on | n | share |
|---|---|---|
| a legible number that is not annotated (mailbox-cluster unit numbers, second house number, numbers on parcels/bins: "8525", "2370", "725", "167", "360", "7449", img263 has 11, img333 has 7) | ~72 | 39% |
| non-numeric text ("WELCOME", "OPEN", "ORD", "U.S. MAIL", "USA", "Hills") | ~39 | 21% |
| non-text pattern (door handles, keypads, vents, doorbells, ornaments, blur) | ~75 | 40% |

Reading: the labels mark *address numbers*, not text. About 4 in 10 of the "false positives" are
numbers that look exactly like positives and are unlabelled by policy or by omission; no weight
edit can remove those without removing true positives. The addressable part is non-numeric text
and non-text patterns, about 114 boxes on val_dev (all protocol), of which only those on non-empty
images count in the fixed measurement.

Note on train: under the eval pipeline a random 800-image train subset scores evalpy H 0.9169 / all
H 0.8939 (P 0.838 on all). The model makes the same kind of "errors" on its own training images, so
train is far more representative than expected, and the 727 empty train images (never trained on:
0.78 boxes per image) behave like held-out negatives.

## 3. H2 — Paddle-to-PyTorch port residue: NEGATIVE

- All 514 Paddle tensors in `best_accuracy.pdparams` are **bit-identical** to the .pth (BN buffers by rename).
- Op attributes in the Paddle inference graph (`inference.json`) match the port: hardsigmoid slope
  1/6 in the 5 backbone SE layers and 0.2 in the 8 neck SE layers, exact GELU (approximate=false).
- Not re-verified: a forward pass in Paddle (no Paddle runtime on this machine).
  `docs/paddle_det_port_parity.md` already reports max diff 5.7e-3 against Paddle CPU kernels.
- The aux heads (`head.aux_*`, 140k parameters) and `head.thresh` are never executed in eval mode
  (their BN layers receive no forward call): confirmed, not touched.

## 4. H1 — BatchNorm statistics mismatch: REAL MISMATCH, RECALIBRATION HARMFUL

`ws/10_bn_stats.py`, `ws/13_bn_seq.py`; logs `logs/bn_stats.txt`, `logs/bn_seq.txt`.
- Measured on 800 train images under the eval pipeline, BN inputs have much lower variance than the
  stored running_var: mean log2 ratio -0.67 over 72 layers, about -2.5 (6x) in the last backbone
  stage and the p5 neck branch, -1.65 at head.binarize.conv_bn1. Most of this is compounding: with
  upstream layers already recalibrated the local mismatch is small except stage 4 (-0.45) and
  `neck.inp_conv_dw.3` (-1.65).
- Recalibration (sequential, layer by layer in execution order, 240 train images, forward only):

| edit | val_dev evalpy P / R / H | paired dH [95% CI] |
|---|---|---|
| original | 0.8355 / 0.9484 / 0.8884 | |
| 25% toward recalibrated | 0.8067 / 0.9484 / 0.8718 | -0.0166 [-0.027, -0.006] |
| 50% | 0.7485 / 0.9324 / 0.8304 | -0.0580 [-0.075, -0.041] |
| 75% | 0.6809 / 0.8993 / 0.7750 | -0.1134 [-0.134, -0.093] |
| 100% | 0.6528 / 0.8845 / 0.7512 | -0.1372 [-0.161, -0.115] |

  Monotone dose-response in the wrong direction: detections rise from 924 to 1103. The stored
  statistics carry the scale of the training crops (text-dense: CopyPaste, zoom); normalising by the
  smaller eval-image variance amplifies text features and the model over-detects. The mismatch is a
  prior the model relies on, not a fault. (A first attempt that updated all layers at once diverged;
  it is not a valid test and is superseded by the sequential one.)
- Decision: rejected.

## 5. H5 — dead / saturated / redundant units: nothing to repair (`ws/14_census.py`, `logs/census.txt`)

- Stage 4 has many dead channels (running_var ~ 0 and gamma ~ 0): 199/768 and 491/768 in the two
  expand layers, 40-88 of 384 elsewhere. Their eval-time output variance is 0 to 1e-5 against 0.02-0.3
  for live channels: truly dead, no noise amplification through eps. Wasted capacity, not a fault.
- SE gates: none stuck. `neck.ins_conv.3` (p5 lateral) is the most saturated (31% at 0, 10% at 1);
  4+6 of 384 channels in `blocks_s4.1` are always 0 / always 1.
- RepLK branches: all four carry energy (smallest share 7.5%).

## 6. H3/H6 — where is the TP-vs-FP information, and can the final layer use it?

Box-level (`ws/20_capture_boxfeat.py`, `ws/21_linear_sep.py`): features mean-pooled over each
detected box; LDA fitted on 3,139 train boxes (2,361 TP, 778 FP), AUC measured on val_dev boxes.

| score | AUC on val_dev (TP vs background FP) |
|---|---|
| model box score (mean probability, what box_thresh sees) | 0.795 |
| mean logit in the box | 0.855 |
| log(box area) alone | 0.815 |
| LDA on h2 (24 ch, input of the last layer) | 0.913-0.919 |
| LDA on h1 (24 ch) | 0.921 |
| LDA on neck output (96 ch) | 0.933 (0.951 for FP on non-empty images) |

Two readings: (a) the box score is a poor ranking statistic because the logits are saturated (TP
boxes mean logit 8.0, FP boxes 3.0; both map to probability ~1), and (b) the features hold more
than the logit uses.

E1 (`ws/23_e1_dose.py`): add the h2 LDA direction to `head.binarize.conv3` with strength alpha.
| alpha (bias: TP mean unchanged) | val_dev evalpy H | dH [95% CI] |
|---|---|---|
| 0.1 | 0.8889 | +0.0005 [-0.002, +0.003] |
| 0.25 | 0.8926 | +0.0042 [+0.001, +0.008] |
| 0.5 | 0.8740 | -0.0144 [-0.033, +0.001] |
| 1.0 | 0.7175 | -0.171 |
| 2.0 | 0.1670 | -0.721 |
Beyond a small dose the edit creates new boxes in the background (924 -> 1257 detections at alpha 1):
the direction was fitted on boxes only and is unconstrained on background pixels. Rejected.

Pixel-level constrained least squares on the same layer (`ws/30_pixstats.py`, `ws/ls_edit.py`):
minimise the change on TP and background pixels while lowering active FP pixels by Delta. Result per
unit Delta, for every weighting tried: FP-active pixels move -0.47, TP-active pixels move -0.53.
**At the input of the last layer, active FP pixels cannot be lowered without lowering TP pixels by
the same amount: a closed-form refit of the final layer degenerates into a bias shift.** Rejected
without a val evaluation (H6 negative).
The same analysis on the neck output (96 ch): FP -0.45 vs TP -0.16 (ratio 3:1), pixel-level
Mahalanobis distance TP-active vs FP-active 3.0 at the neck against 1.6 at h2. The distinction
exists at the neck and is discarded by head.binarize.conv1.

## 7. Baseline B — global knobs on the last layer (`ws/32_baselineB.py`, `logs/baselineB_dev.txt`)

logit' = s * (logit - L0) + L0 + delta with L0 = logit(0.2): `conv3.weight *= s`,
`conv3.bias = s*(bias - L0) + L0 + delta`. With delta = 0 the thresh-0.2 contour, hence every box,
is unchanged; only the box scores are un-saturated.

| s (delta 0) | val_dev fixed pp, evalpy H / all H | best pp on the grid, evalpy H |
|---|---|---|
| 1 (original) | 0.8884 / 0.8413 | 0.9044 |
| 0.7 | 0.8920 / 0.8464 | 0.9066 |
| 0.5 | 0.8961 / 0.8527 | 0.9117 |
| 0.35 | 0.9032 / 0.8631 | 0.9164 |
| 0.25 | 0.9128 / 0.8764 | 0.9215 |
| 0.18 | 0.9083 / 0.8842 | 0.9180 |
| 0.12 | 0.7630 / 0.7504 | 0.9210 |
A plain bias shift (s = 1, delta = -1) gives 0.8904 at fixed pp and 0.9046 at best pp: equivalent to
moving thresholds, as expected. **Scaling is not equivalent to moving thresh/box_thresh**: it
changes the box score from a saturated mean probability into something close to the mean logit,
and it beats the postprocess-only ceiling by +0.017 hmean on val_dev (0.9215 vs 0.9044). By the
task's definition this is still a global knob (baseline B), so it is the bar for targeted edits.
