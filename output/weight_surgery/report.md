# Weight surgery on the PP-OCRv6 small detector — report

## Result

**One edit qualified and is frozen as `output/weight_surgery/best_edited.pth`: the logits of the last layer of
the binarize head are scaled by 0.25 about the thresh-0.2 contour.** It raises hmean on every split at the
fixed postprocess, including the test set. By the task's own definition it is a **global knob (baseline B), not
surgery**. One genuinely targeted edit (E2, the head's reading of nine neck channels) has the predicted causal
effect and the right sign everywhere, but on held-out data its gain under the fixed measurement is inside the
noise, and it adds nothing on top of the knob. It is not in the frozen checkpoint.

Two things found along the way matter more than either edit:

1. **`tools/eval.py` never scores images without labels**, and the training loader never trained on them.
   61% of the test set is such images.
2. **About 40% of the "false positives" on val are legible numbers that are simply not annotated.**

### Headline: test set (2,990 images, 1,837 without text; never used for any choice)

Run once, after the choice was frozen in `notes.md` (section 12). Precision / recall / hmean.

| | all images scored (empty images count) | as `tools/eval.py` counts (empty images skipped) | boxes on the 1,837 empty images |
|---|---|---|---|
| original, fixed postprocess | 0.3168 / 0.8903 / **0.4673** | 0.7751 / 0.8855 / **0.8266** | 2,083 |
| original, best postprocess A* | 0.3951 / 0.8823 / 0.5458 | 0.8347 / 0.8811 / 0.8573 | 1,465 |
| **edited (B), fixed postprocess** | 0.4468 / 0.8775 / **0.5921** | 0.8567 / 0.8721 / **0.8643** | 1,175 |
| edited (B), at A* | 0.8663 / 0.6173 / 0.7209 | 0.9607 / 0.6164 / 0.7509 | 93 |
| *secondary (1280-max, box_thresh 0.6)*: original | 0.3732 / 0.8823 / 0.5245 | 0.8186 / 0.8749 / 0.8458 | 1,614 |
| *secondary*: edited (B) | 0.7165 / 0.7830 / 0.7483 | 0.9351 / 0.7800 / 0.8506 | 316 |

Paired bootstrap over images (10,000 resamples), difference in hmean with 95% interval:

| comparison on test | all images | `tools/eval.py` count |
|---|---|---|
| edited vs original, both at fixed postprocess | **+0.1248 [+0.1161, +0.1338]** | **+0.0377 [+0.0284, +0.0473]** |
| postprocess alone (original at A* vs original at fixed) | +0.0785 [+0.0709, +0.0862] | +0.0307 [+0.0191, +0.0439] |
| edited at fixed vs original at A* | +0.0463 [+0.0393, +0.0532] | +0.0071 [-0.0060, +0.0178] |
| edited at A* vs original at A* | +0.1751 [+0.1407, +0.2076] | -0.1063 [-0.1367, -0.0773] |
| secondary setting, edited vs original | +0.2238 [+0.2027, +0.2460] | +0.0047 [-0.0123, +0.0210] |

How to read it:

- At the fixed postprocess the edit is a clear gain on test under both counts, at a recall cost of 1.3 points.
- Against the best postprocess-only setting, the edit is clearly better when empty images are counted
  (+0.046) and indistinguishable under the `tools/eval.py` count (+0.007, interval spans zero). On val the same
  comparison was +0.011 and clear of zero (below), so this part of the val result did not fully carry over.
- "Both combined" does not work for this edit, as predicted before the test: the edit rescales the score axis
  and is matched to box_thresh 0.45. At box_thresh 0.65 (A*) or 0.6 (secondary) recall falls to 0.62 / 0.78.
  Those rows have the highest all-images hmean only because the test set is 61% empty, so any setting that
  trades recall for precision wins there; under the `tools/eval.py` count the same rows lose 0.11 hmean.
  **If this checkpoint is shipped, box_thresh must stay near 0.45.**
- Test precision with every image counted is 0.32 for the original and 0.45 after the edit. No weight edit
  found here fixes that: the model still puts 1,175 boxes on 878 text-free images.

### All splits, fixed postprocess (thresh 0.2, box_thresh 0.45, unclip 1.4)

P / R / H as `tools/eval.py` counts; all-images hmean in brackets. Intervals are paired bootstrap vs the original.

| split | role | original | edited (B) | dH `tools/eval.py` count | dH all images |
|---|---|---|---|---|---|
| val_dev (784) | **used to choose** | 0.8355 / 0.9484 / 0.8884 (0.8413) | 0.8884 / 0.9386 / 0.9128 (0.8764) | +0.0244 [+0.0157, +0.0336] | +0.0352 [+0.0256, +0.0451] |
| val_holdout (769) | held out | 0.8570 / 0.9322 / 0.8930 (0.8416) | 0.9084 / 0.9210 / 0.9146 (0.8822) | +0.0216 [+0.0124, +0.0313] | +0.0406 [+0.0294, +0.0523] |
| full val (1,553) | mixed | 0.8490 / 0.9434 / 0.8937 (0.8414) | 0.8993 / 0.9323 / 0.9155 (0.8792) | +0.0217 [+0.0153, +0.0282] | +0.0378 [+0.0302, +0.0454] |
| test (2,990) | **held out, final** | 0.7751 / 0.8855 / 0.8266 (0.4673) | 0.8567 / 0.8721 / 0.8643 (0.5921) | +0.0377 [+0.0284, +0.0473] | +0.1248 [+0.1161, +0.1338] |

`tools/eval.py` itself on full val: original 0.8499 / 0.9434 / 0.8943, edited **0.8993 / 0.9323 / 0.9155**.

### The three numbers asked for, per candidate (hmean, `tools/eval.py` count)

A* = thresh 0.10, box_thresh 0.65, unclip 1.3, the best of 560 settings on val_dev.

| | val_dev (chosen on) | val_holdout | full val | test |
|---|---|---|---|---|
| best postprocess alone (original at A*) | 0.9044 | 0.9037 | 0.9049 | 0.8573 |
| B at fixed postprocess | 0.9128 | 0.9146 | 0.9155 | 0.8643 |
| B at A* (literal "both combined") | 0.8176 | 0.8198 | 0.8173 | 0.7509 |
| B at fixed vs original at A*, dH | +0.0084 [-0.0002, +0.0164] | +0.0109 [+0.0049, +0.0177] | +0.0106 [+0.0057, +0.0159] | +0.0071 [-0.0060, +0.0178] |
| E2 (targeted) at fixed postprocess | 0.8969 | 0.8977 | 0.8996 | not run |
| E2 at fixed vs original at fixed, dH | +0.0085 [+0.0011, +0.0158] | +0.0047 [-0.0043, +0.0135] | +0.0059 [-0.0003, +0.0119] | |
| E2 at A* ("both combined") | 0.9166 | 0.9084 | 0.9140 | not run |
| E2 at A* vs original at A*, dH | +0.0122 [+0.0052, +0.0200] | +0.0047 [-0.0027, +0.0120] | +0.0092 [+0.0037, +0.0151] | |
| B + E2 at fixed postprocess | 0.9194 | 0.9127 | 0.9173 | not run |
| B + E2 vs B, dH | +0.0067 [+0.0011, +0.0126] | -0.0019 [-0.0096, +0.0054] | +0.0019 [-0.0030, +0.0066] | |

Counting all images, E2 alone is clear of zero on the holdout too: +0.0144 [+0.0043, +0.0248].

**Holdout evaluations used: 3 of 5** (E2, B, B+E2), each at settings fixed beforehand, no sweep.

---

## What the measurement actually counts

`DetLabelEncode` returns `None` for an empty label list, and `SimpleDataSet.__getitem__` then substitutes
sample `(idx + 1) % N`. So in `tools/eval.py`:

- val.txt has 184 empty-label images out of 1,553. They are never scored; the image after each one is scored
  twice. The reported "1,553 images" are 1,369 distinct ones.
- The original model puts 171 boxes on those 184 images. None of it shows in the 0.8943.
- On the test set this skips all 1,837 empty images and counts some labelled images many times over
  (3,214 ground-truth boxes are tallied from 1,249).

Training used the same loader (PaddleOCR's behaves identically), so **the 727 empty-label images in train.txt
were never trained on.** The model puts 0.78 boxes per image on them, against 0.93 on val's empty images and 1.13 on the test set's.

Every number in this report is therefore given twice: as `tools/eval.py` counts, and with every image scored.
The harness (`ws/common.py`) reproduces `tools/eval.py` to within 2 detections out of 1,788 on full val; that
difference is GPU numerics between processes and is the noise floor (about 0.0006 hmean).

## The edit that was kept: B, un-saturating the box score

**Tensors changed (2 tensors, 97 values; names, shapes, dtypes unchanged):**

| tensor | change |
|---|---|
| `head.binarize.conv3.weight` (24, 1, 2, 2) | multiplied by 0.25 |
| `head.binarize.conv3.bias` (1,) | `0.25 * (bias - L0) + L0` with L0 = ln(0.2 / 0.8) = -1.3863: 0.1132 -> -1.0114 |

The new logit is `0.25 * (logit - L0) + L0`. The thresh-0.2 contour is unchanged, so every region and every
box is exactly what it was; only the box *score* changes.

**Why it helps.** DBPostProcess keeps a box if the mean probability inside it exceeds box_thresh. The model's
logits are saturated: on train boxes the mean logit is 8.0 for true positives and 3.0 for false positives, and
the sigmoid maps both to about 1. So the score cannot rank them:

| score, on val_dev boxes | AUC, true positive vs background false positive |
|---|---|
| mean probability (what box_thresh sees) | 0.795 |
| mean logit | 0.855 |

Compressing the logits makes the mean probability behave like the mean logit. That is why this is not
equivalent to moving thresh or box_thresh, and why it beats the postprocess sweep. A plain bias shift, by
contrast, *is* equivalent to moving thresholds (val_dev 0.8904 at fixed, 0.9046 at best postprocess).

**Dose-response (val_dev, fixed postprocess; boxes identical at every dose):**

| scale s | 1 | 0.7 | 0.5 | 0.4 | 0.35 | 0.30 | 0.27 | **0.25** | 0.23 | 0.20 | 0.18 | 0.12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hmean, `tools/eval.py` count | 0.8884 | 0.8920 | 0.8961 | 0.8969 | 0.9032 | 0.9078 | 0.9102 | **0.9128** | 0.9148 | 0.9161 | 0.9083 | 0.7630 |
| hmean, all images | 0.8413 | 0.8464 | 0.8527 | 0.8571 | 0.8631 | 0.8704 | 0.8738 | **0.8764** | 0.8801 | 0.8862 | 0.8842 | 0.7504 |

Smooth rise to a plateau at 0.20-0.27, then collapse once true positives fall under box_thresh. 0.25 was taken
as the middle of the plateau. Cost: recall -1.0 point on val, -1.3 on test.

**What it removes.** Of the 186 background false positives I tallied by hand on val_dev, B removes 41 of 75
non-text patterns, 11 of 39 non-numeric texts and 10 of 72 unannotated numbers.

## The targeted edit that was not kept: E2, nine neck channels as read by the head

**Tensor changed:** `head.binarize.conv1.weight` (24, 96, 3, 3). Input channels 79, 83, 80, 66, 88 multiplied by
0; input channels 38, 77, 68, 50 multiplied by 2. Relative L2 change of the tensor 0.38.
Checkpoints: `cand_E2_sup5x0_amp4x2.pth`, and with B: `cand_BE2_s0.25_sup5x0_amp4x2.pth`.

**Localisation.** Where is the true-vs-false-positive distinction available?
- At the input of the last layer it is not, pixel by pixel. A closed-form least-squares edit of that layer that
  holds true-positive and background pixels fixed while lowering active false-positive pixels moves them
  together: FP -0.47, TP -0.53 per unit. A refit of the final layer degenerates into a bias shift.
- At the neck output it is: the same least-squares problem gives FP -0.45, TP -0.16. The distinction exists in
  the 96 neck channels and is discarded by `head.binarize.conv1`.

**Causal scan (train images only: 700 images, 420 true-positive and 294 false-positive boxes).** Each neck
channel's weights into `conv1` were set to 0 or doubled and the change of the mean logit inside baseline boxes
measured, re-running only the head.

| channel (FPN level) | ablate: TP | ablate: FP | double: TP | double: FP |
|---|---|---|---|---|
| 79 (p2) | +0.03 | **-0.87** | -0.23 | +0.05 |
| 83 (p2) | +0.01 | -0.33 | -0.02 | +0.27 |
| 80 (p2) | +0.01 | -0.17 | -0.01 | +0.15 |
| 66 (p3) | +0.00 | -0.14 | -0.01 | +0.10 |
| 88 (p2) | -0.00 | -0.12 | -0.00 | +0.09 |
| 38 (p4) | +0.03 | +0.28 | -0.06 | -0.37 |
| 77 (p2) | -0.01 | +0.25 | +0.00 | -0.30 |
| 68 (p3) | +0.03 | +0.30 | -0.04 | -0.33 |
| 50 (p3) | +0.03 | +0.32 | -0.04 | -0.36 |

All nine together, on the train lab: TP -0.21, FP -3.44; AUC of the mean logit 0.8885 -> 0.9245.
Control: 20 random sets of channels with the same edit shape give AUC 0.878 on average (range 0.742-0.916).

**Dose-response of channel 79 alone (val_dev, fixed postprocess, original score scale):**

| gain on channel 79 | 2 | 1.5 | 1 | 0.5 | 0 | -0.5 | -1 |
|---|---|---|---|---|---|---|---|
| hmean, `tools/eval.py` count | 0.8844 | 0.8880 | 0.8884 | 0.8904 | 0.8921 | 0.8952 | 0.8970 |
| dH vs original | -0.0039 | -0.0004 | | +0.0020 | +0.0037 | +0.0068 | +0.0087 |
| 95% interval | [-0.008, -0.001] | [-0.003, +0.002] | | [-0.001, +0.005] | [-0.001, +0.009] | [+0.002, +0.013] | [+0.001, +0.017] |

Monotone in the predicted direction, and it reverses when the channel is amplified. Recall is unchanged down
to gain -0.5.

**Why it was not kept.** On the holdout E2 alone gives +0.0047 [-0.0043, +0.0135] under the fixed measurement
(+0.0144 [+0.0043, +0.0248] counting all images), and on top of B it gives -0.0019 [-0.0096, +0.0054]. E2 and B
remove largely the same low-logit boxes, and B does it better. The mechanism is real; the gain is not large
enough to separate from noise with 769 images.

Limits of the evidence: the nine channels were chosen on train, but the particular combination was picked among
11 variants looked at on val_dev. And the mean activation of channel 79 does not by itself distinguish the
groups (true positives +3.4 sd, non-text false positives +2.4 sd): the claim is about what the head does with
the channel, not that the channel "fires on false positives".

## Baseline A: postprocess sweep

560 settings on val_dev (thresh 0.1-0.5, box_thresh 0.3-0.8, unclip 1.0-2.0). The gain is almost all
box_thresh; unclip is flat between 1.3 and 1.5 and thresh moves hmean by under 0.005.

| | val_dev (chosen on) | val_holdout | test |
|---|---|---|---|
| original at A* vs at fixed, dH `tools/eval.py` count | +0.0160 [+0.0069, +0.0262] | +0.0107 [+0.0019, +0.0196] | +0.0307 [+0.0191, +0.0439] |
| same, all images | +0.0226 [+0.0133, +0.0322] | +0.0243 [+0.0146, +0.0343] | +0.0785 [+0.0709, +0.0862] |

## Error taxonomy (val_dev, original model, fixed postprocess, all 784 images)

910 detections, 689 matched, 221 unmatched. 728 ground-truth boxes, 39 unmatched.

| unmatched detections | n |
|---|---|
| box on something the annotation does not mark, on an image that has labels | 102 |
| box on an empty-label image (invisible to `tools/eval.py`) | 84 |
| split: several boxes inside one ground-truth box | 16 |
| too tight (IoU 0.3-0.5, box inside the ground truth) | 7 |
| fragment inside a ground-truth box, IoU < 0.3 | 4 |
| too loose | 6 |
| low overlap | 2 |
| merge of several ground-truth boxes | 0 |

| unmatched ground truth | n |
|---|---|
| missed entirely | 18 |
| best box at IoU 0.3-0.5 | 10 |
| best box under IoU 0.3 | 9 |
| split | 5 |

Matched boxes have mean IoU 0.851 and only 2.5% sit between 0.5 and 0.6: geometry is not the problem.
Splits, merges and loose boxes are 35 of 221; I did not pursue them.

**186 of the 221 unmatched detections (84%) are background boxes.** I looked at all 186 crops
(`viz/fp_all/sheet_0.jpg` to `sheet_5.jpg`; tally in `fp_manual_tally_dev.json`):

| what the box is on | all 186 | the 102 on labelled images |
|---|---|---|
| a legible number that is not annotated | 72 (39%) | 43 (42%) |
| non-numeric text ("WELCOME", "OPEN", "ORD", "U.S. MAIL") | 39 (21%) | 23 (23%) |
| a non-text pattern (door handles, keypads, vents, doorbells, blur) | 75 (40%) | 36 (35%) |

The unannotated numbers are mailbox-cluster unit numbers (one image has 11, another 7), second house numbers,
numbers on bins and parcels: "8525", "2370", "725", "167", "360". They look exactly like the positives. Whether
this is label policy or omission I cannot tell from the data, but either way **no weight edit can remove them
without removing true positives**, and the two edits here indeed leave 86-89% of them in place (75% when combined). The model
makes the same errors on its own training images (800 train images under the eval pipeline: precision 0.84
with all images counted), which is what an inconsistent target looks like.

## Hypotheses that failed

| hypothesis | result |
|---|---|
| BatchNorm statistics mismatch | The mismatch is real: eval-time variance is up to 6x below the stored running variance in the last backbone stage and the p5 neck branch. Recalibrating on train images is harmful with a monotone dose-response: 25 / 50 / 75 / 100% toward recalibrated gives dH -0.017 / -0.058 / -0.113 / -0.137 on val_dev, detections rising from 924 to 1,103. The stored statistics carry the scale of the text-dense training crops and the model relies on it. |
| Paddle-to-PyTorch port residue | None found. All 514 Paddle tensors are bit-identical to the `.pth`. Hardsigmoid slopes (1/6 in 5 backbone SE layers, 0.2 in 8 neck SE layers) and exact GELU match the Paddle inference graph. |
| Dead, saturated or redundant units | Stage 4 has many dead channels (199/768 and 491/768 in its two expand layers) but they output a constant: wasted capacity, nothing to repair. No SE gate is stuck; all four RepLK branches carry energy. |
| Closed-form refit of the final layer | Degenerates into a bias shift (see E2, localisation). A box-level LDA direction added to the final layer gives +0.004 at low strength and then creates new background boxes (924 -> 1,257 detections). |
| Aux heads | `head.aux_*` (140k parameters) and `head.thresh` are never executed in eval mode. Not touched. |

## Not verified

- **Export.** `tools/deployment/export_onnx.py` could not run here: the config has no `Export` section, and with
  one supplied the exporter fails on a missing `onnxscript` package, for any checkpoint. The edited checkpoint
  loads with `strict=True` and changes only the values of two existing tensors, so I expect MNN and CoreML
  export to behave as for the original, but that is untested. Worth checking on device with the fp16 CoreML build,
  since the edit changes the logit range by 4x.
- **Paddle forward parity** was not re-run (no Paddle runtime); the weight and graph checks above stand in for it.
- **E2 and B+E2 on the test set**: deliberately not run, to keep the test to the one frozen candidate.
- Whether the unannotated numbers are policy or omission.

## Where the larger gains are

Weight editing bought +0.022 hmean on val and +0.038 on test under the fixed measurement, from a knob. The
evidence points elsewhere for anything bigger:

1. **Train on the empty images.** The loader drops them, 727 in train alone. On test the model puts 2,083 boxes
   on 1,837 text-free images and precision with every image counted is 0.32; after the edit it is still 0.45.
   This is the largest gap in the whole study and no edit closes it. Fixing `DetLabelEncode` / `SimpleDataSet`
   so empty images are kept, then fine-tuning, is the first thing to try.
2. **Make `tools/eval.py` score empty images**, so the number being optimised sees them. The checkpoint was
   selected on a metric blind to 12% of val and 61% of test.
3. **Relabel or state the policy** for non-address numbers. They are 42% of the false positives the fixed
   measurement counts. With every other false positive removed and nothing else changed, val_dev precision
   would be about 0.90, not 1.0.
4. **If nothing is retrained, ship the knob, or get most of it for free:** on val, postprocess alone
   (box_thresh 0.65) gives +0.011 and the edit +0.022. The edit is tied to box_thresh 0.45.

## Files

- `best_edited.pth` — frozen checkpoint (B). `cand_E2_*.pth`, `cand_BE2_*.pth` — the targeted edit, alone and with B.
- `notes.md` — running log, including the decision rule and the frozen choice written before the test.
- `ws/` — all scripts; `logs/` — raw outputs; `viz/` — crop sheets; `val_dev.txt`, `val_holdout.txt` — the split.
- `weights/best_accuracy.pth` is untouched (md5 33db92a5910fa568724bfd1a278eeae1).
