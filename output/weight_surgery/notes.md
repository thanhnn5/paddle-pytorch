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

Finer s grid at fixed pp (`ws/33_B_fine.py`): plateau for s in [0.20, 0.27], val_dev evalpy H 0.9102-0.9161,
paired dH vs original +0.022 to +0.028, all intervals above +0.013. s = 0.25 taken (middle of the plateau):
evalpy P 0.8884 R 0.9386 H 0.9128, dH +0.0244 [+0.0157, +0.0336]; all H 0.8764, dH +0.0352 [+0.0256, +0.0451].
Caveat: B rescales the score axis, so it is tied to box_thresh 0.45. At A* (box_thresh 0.65) recall collapses.

## 8. H3 — causal scan of the neck channels read by the head (`ws/headlab.py`, `ws/40_channel_scan.py`)

Head-only lab: neck outputs of 700 train images cached (300 empty-label, 400 with labels; 420 TP boxes,
294 FP boxes). For each of the 96 neck output channels, `head.binarize.conv1.weight[:, c]` is multiplied by
0 (ablate) or 2 and the change of the mean logit inside baseline boxes is measured (`logs/channel_scan.txt`).

Most FP-selective channels (ablation): dTP / dFP in logits
| channel (level) | ablate dTP | ablate dFP | background pixels above contour | double dTP | double dFP |
|---|---|---|---|---|---|
| 79 (p2) | +0.03 | -0.87 | -1387 | -0.23 | +0.05 |
| 83 (p2) | +0.01 | -0.33 | -369 | -0.02 | +0.27 |
| 80 (p2) | +0.01 | -0.17 | -314 | -0.01 | +0.15 |
| 66 (p3) | +0.00 | -0.14 | -394 | -0.01 | +0.10 |
| 88 (p2) | -0.00 | -0.12 | -120 | -0.00 | +0.09 |
FP-suppressing channels (doubling lowers FP, TP unchanged): 38 (p4) -0.06/-0.37, 77 (p2) 0.00/-0.30,
68 (p3) -0.04/-0.33, 50 (p3) -0.04/-0.36.

Dose-response of channel 79 on val_dev, fixed pp, original score scale (`ws/41_e2_dose.py`):
| gain on ch 79 | evalpy H | dH [95% CI] | all H |
|---|---|---|---|
| 2 (reverse) | 0.8844 | -0.0039 [-0.0076, -0.0005] | 0.8369 |
| 1.5 (reverse) | 0.8880 | -0.0004 [-0.003, +0.002] | 0.8404 |
| 1 (original) | 0.8884 | | 0.8413 |
| 0.5 | 0.8904 | +0.0020 [-0.0005, +0.005] | 0.8444 |
| 0 | 0.8921 | +0.0037 [-0.0005, +0.009] | 0.8466 |
| -0.5 | 0.8952 | +0.0068 [+0.0015, +0.013] | 0.8529 |
| -1 | 0.8970 | +0.0087 [+0.0006, +0.017] | 0.8568 |
Monotone in the predicted direction, reversed sign when amplified; recall unchanged up to gain -0.5.

**E2** = gain 0 on channels 79, 83, 80, 66, 88 and gain 2 on channels 38, 77, 68, 50 (channels chosen on
train, strengths 0 / 2 chosen as the plain "ablate / double"; the combination was picked among 11 variants
looked at on val_dev, so its dev numbers are optimistic).
- Train lab: dTP -0.21, dFP -3.44, AUC(mean logit) 0.8885 -> 0.9245. Control: 20 random channel sets with
  the same edit shape give AUC 0.878 on average (min 0.742, max 0.916) and dFP-dTP +0.10 on average.
- Mean pooled activation of channel 79 does not itself separate the groups (TP +3.4 sd, non-text FP +2.4 sd):
  the claim is causal (what the head does with the channel), not "the channel fires on false positives".

## 9. Candidates on val_dev (`ws/50_make_candidates.py`, `ws/51_cand_dev.py`, `logs/cand_dev.txt`)

A* = best postprocess from baseline A on val_dev = (thresh 0.10, box_thresh 0.65, unclip 1.3).
evalpy hmean (all-images hmean in brackets); everything here was used for selection.

| checkpoint | fixed pp | at A* | own best pp on dev |
|---|---|---|---|
| original | 0.8884 (0.8413) | 0.9044 (0.8638) | 0.9044 @ A* |
| E2 (targeted) | 0.8969 (0.8561) | 0.9166 (0.8814) | 0.9166 @ A* |
| B (s = 0.25, knob) | 0.9128 (0.8764) | 0.8176 (0.8013) | 0.9215 @ (0.1, 0.5, 1.3) |
| BE2 (both) | 0.9194 (0.8895) | 0.7977 (0.7807) | 0.9237 @ (0.1, 0.4, 1.4) |

Paired bootstrap on val_dev (evalpy):
- E2 vs original, fixed pp: dH +0.0085 [+0.0011, +0.0158]; both at A*: dH +0.0122 [+0.0052, +0.0200].
  So E2 beats the unedited model at the fixed postprocess and adds on top of the best postprocess.
- BE2 vs B, fixed pp: dH +0.0067 [+0.0011, +0.0126] (all: +0.0130 [+0.0061, +0.0205]); at B's best pp the
  difference is inside the noise (-0.008 [-0.018, +0.000] at (0.1, 0.5, 1.3)). E2 and B remove largely the
  same low-logit boxes; once the score is un-saturated and the threshold tuned, E2 adds little.
- Which of the 186 hand-tallied baseline FPs survive at fixed pp:
  | | unannotated numbers (72) | letters (39) | non-text (75) |
  |---|---|---|---|
  | B | 62 | 28 | 34 |
  | E2 | 64 | 32 | 49 |
  | BE2 | 54 | 23 | 21 |
  Both edits remove mostly non-text patterns (BE2: 72% of them) and few of the unannotated numbers (25%).

## 10. Decision rule for the holdout (written before any holdout evaluation)

Three holdout evaluations (E2, B, BE2), each checkpoint scored only at settings fixed in advance:
the fixed pp; A* = (0.10, 0.65, 1.3); and its own dev-best pp (E2: A*; B: (0.1, 0.5, 1.3); BE2: (0.1, 0.4, 1.4)).
No sweep on the holdout. An edit qualifies if its holdout hmean (evalpy, fixed pp) is above the original's.
Frozen candidate: BE2 if E2 qualifies and BE2 >= B on the holdout at fixed pp; else B if B qualifies;
else E2 if E2 qualifies; else none.

## 11. Holdout evaluations (3 of 5 used) — `ws/60_holdout.py`, `logs/holdout.txt`

val_holdout: 769 images, 99 empty. Settings fixed in advance, no sweep. evalpy P / R / H (all-images H).

| checkpoint | fixed pp | paired dH vs original, fixed pp [95% CI] | at A* | own dev-best pp |
|---|---|---|---|---|
| original | 0.8570 / 0.9322 / 0.8930 (0.8416) | | 0.8894 / 0.9184 / 0.9037 (0.8659) | |
| E2 (holdout eval #1) | 0.8768 / 0.9197 / 0.8977 (0.8560) | +0.0047 [-0.0043, +0.0135]; all +0.0144 [+0.0043, +0.0248] | 0.9022 / 0.9147 / 0.9084 (0.8765) | = A* |
| B (holdout eval #2) | 0.9084 / 0.9210 / 0.9146 (0.8822) | +0.0216 [+0.0124, +0.0313]; all +0.0406 [+0.0294, +0.0523] | 0.9741 / 0.7077 / 0.8198 | (0.1, 0.5, 1.3): 0.9062 (0.8826) |
| BE2 (holdout eval #3) | 0.9133 / 0.9122 / 0.9127 (0.8859) | +0.0197 [+0.0084, +0.0313]; all +0.0444 [+0.0306, +0.0587] | 0.9790 / 0.6424 / 0.7758 | (0.1, 0.4, 1.4): 0.9184 (0.8911) |

- Postprocess alone (A*) on the holdout: +0.0107 hmean (0.8930 -> 0.9037), down from +0.0160 on dev.
- E2 at A* vs original at A*: +0.0047 [-0.0027, +0.0120] evalpy, +0.0106 [+0.0019, +0.0195] all.
  E2 has the right sign everywhere but the fixed-measurement gain is inside the noise on held-out data.
- BE2 vs B at fixed pp: -0.0019 [-0.0096, +0.0054] evalpy, +0.0038 [-0.0053, +0.0126] all. No added value.
- B's dev-tuned postprocess did not transfer (0.9062 < 0.9146 at fixed pp): for B the fixed pp is its setting.

All three improve holdout hmean at the fixed postprocess, so all three "qualify" by the letter of the rule;
only B (and BE2, through B) is clear of the noise.

## 12. FROZEN CHOICE (written before any contact with the test set)

By the rule of section 10 (BE2 only if BE2 >= B on the holdout at fixed pp; it is not: 0.9127 < 0.9146):

- **Frozen checkpoint: B** = `output/weight_surgery/cand_B_s0.25.pth`, copied to `best_edited.pth`.
  Edit: `head.binarize.conv3.weight *= 0.25`, `head.binarize.conv3.bias = 0.25*(bias - L0) + L0`,
  L0 = ln(0.2/0.8) = -1.3863. Two tensors, 97 values. This is a global knob (baseline B class), not surgery.
- **Frozen best postprocess from baseline A: A* = thresh 0.10, box_thresh 0.65, unclip 1.3.**
- Test plan, each of {original, B} run once per resize protocol:
  1. default test resize (736-min), scored at the fixed pp and at A*;
  2. secondary: 1280-max resize, thresh 0.2, box_thresh 0.6, unclip 1.4.
  Both aggregations (evalpy = what tools/eval.py computes, which skips the empty test images; all = every
  image, the count that matches earlier test evaluations and the "empty images cost precision" concern).
- Expectations stated in advance: B rescales the score axis and is matched to box_thresh 0.45, so B at A*
  (box_thresh 0.65) and B at the secondary setting (box_thresh 0.6) should lose a lot of recall; the
  meaningful comparisons are B at fixed pp vs original at fixed pp, and B at fixed pp vs original at A*.
  Because 61% of test images are empty and B removes mostly non-text boxes, the all-images gain on test
  should be at least as large as on val; the evalpy gain should be similar to val (+0.02).
- Nothing is changed after the test numbers. E2 and BE2 are not run on test.
- Holdout evaluations used: 3 / 5.

## 13. FINAL TEST (run once, after the freeze) — `ws/70_test.py`, `logs/test.txt`

2,990 images, 1,837 empty. P / R / H.
| | all images | evalpy | boxes on empty images |
|---|---|---|---|
| original, fixed pp | 0.3168 / 0.8903 / 0.4673 | 0.7751 / 0.8855 / 0.8266 | 2083 |
| original, A* | 0.3951 / 0.8823 / 0.5458 | 0.8347 / 0.8811 / 0.8573 | 1465 |
| B, fixed pp | 0.4468 / 0.8775 / 0.5921 | 0.8567 / 0.8721 / 0.8643 | 1175 |
| B, A* | 0.8663 / 0.6173 / 0.7209 | 0.9607 / 0.6164 / 0.7509 | 93 |
| secondary 1280-max bt 0.6, original | 0.3732 / 0.8823 / 0.5245 | 0.8186 / 0.8749 / 0.8458 | 1614 |
| secondary, B | 0.7165 / 0.7830 / 0.7483 | 0.9351 / 0.7800 / 0.8506 | 316 |

Paired dH: B vs original at fixed pp: all +0.1248 [+0.1161, +0.1338], evalpy +0.0377 [+0.0284, +0.0473].
B at fixed vs original at A*: all +0.0463 [+0.0393, +0.0532], evalpy +0.0071 [-0.0060, +0.0178].
As expected beforehand: B at A* and at box_thresh 0.6 loses recall (0.62 / 0.78). Those rows score highest on
the all-images count only because 61% of test is empty. Nothing was changed after these numbers.

Also after the freeze (no selection): full val for all candidates (`logs/val_full.txt`); `tools/eval.py` on
`best_edited.pth`, full val: P 0.8993 R 0.9323 H 0.9155 (original 0.8499 / 0.9434 / 0.8943).
Export check not possible here (config lacks `Export`; exporter needs `onnxscript`, not installed).
Report: `report.md`. Holdout evaluations used: 3 / 5.
