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
