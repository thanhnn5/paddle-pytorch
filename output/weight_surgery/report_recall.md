# Weight surgery for recall on the PP-OCRv6 small detector — report (part 2)

Objective: more recall at little cost in precision, so that the address unit is more likely to be boxed and the
recognition and address-matching stages can sort out the rest. Same rules as part 1 (`report.md`): no gradients,
edits chosen on train and val_dev, held-out confirmation, fixed postprocess (thresh 0.2, box_thresh 0.45, unclip 1.4).

## Result

**No weight edit qualified.** Two candidates reached the holdout and both failed the rule written down beforehand.
`best_edited.pth` is unchanged. A later candidate (addendum below) passed the holdout and then failed on the test set.

| val_holdout (769 images, never used for choosing), fixed postprocess | P / R, all images | boxes gained / lost | recall change [95% CI] | precision change [95% CI] |
|---|---|---|---|---|
| original | 0.7660 / 0.9337 | | | |
| T: stage-4 lateral scaled x0.75 on its top-down route (targeted) | 0.7687 / 0.9337 | +2 / -2 | 0.0000 [-0.006, +0.006] | +0.003 [-0.008, +0.014] |
| K: last-layer scale 0.5, shift +3 (a global knob) | 0.7593 / 0.9366 | +5 / -3 | +0.003 [-0.004, +0.011] | -0.007 [-0.017, +0.004] |

On val_dev, where they were chosen, T was +3 / 0 and K was +5 / 0 (recall +0.4 and +0.7 point).

The reason is structural, and it is the main finding:

1. **Thresholds cannot buy recall.** Over 420 postprocess settings on val_dev, recall moves from 0.9484 to at most
   0.9545 (5 boxes of 814). Missed numbers are not just under the threshold: the model answers logit -5 to -11
   on them, against +8 on detected text.
2. **A third of the counted misses are not misses for this purpose.** Of 42 unmatched ground-truth boxes on
   val_dev, about 14 have the number correctly boxed inside a loosely drawn annotation (or the reverse) and fail
   only the IoU 0.5 test. About 18 are true blanks and about 8 are vertical stacks or widely spaced digits boxed
   digit by digit.
3. **The blanks are a learned decision, and it is made deep in the backbone.** On training images many blanks are
   sharp, legible numbers (mailbox units "36", "38", "39"; "4125", "6613", "Suite #2"). Pooled over the box, they
   stay separable from background at every depth (AUC 0.95-0.98), and the split between reported and unreported
   numbers appears in backbone stages 3-4 (AUC 0.70 after stage 1, 0.996 after stage 4).
4. **That evidence cannot be read out selectively with the capacity downstream.** Per pixel, every readout tried
   lifts background along with the missed text, and blanks are rare (about 2.7 per 100 images) while background
   is everywhere. To gain one point of recall for under one point of precision an edit must recover a third of
   the blanks while adding fewer than about 1.5 boxes per 100 images. The best detector found adds 50.

## What was tried

All fitted on train only; "lab" is every train image with a fully missed box (283 such boxes) plus 600 controls.

| hypothesis | experiment | outcome |
|---|---|---|
| A neck channel suppresses missed text at the head | gain 0 / 2 / 3 on each of the 96 channels read by `head.binarize.conv1` | Best: +2.7 logits on misses (channel 93 doubled) with 99,000 new background pixels over the contour; +9 is needed. Rejected. |
| The last layer ignores a direction that marks missed text | closed-form ridge refit of `head.binarize.conv3` on the 24 hidden channels | Background moves +0.23 to +0.50 per +1.0 on missed text. On the fit images: 13 misses recovered for 59 new boxes. Rejected without a val run. |
| A thresholded detector on the neck could be wired into the head | Fisher direction on the 96 neck channels (centre + 3x3 mean), fitted on half the lab, scored on the other half | 34% of blanks reached at 50 background blobs per 100 images. Rejected. |
| Stage 4 suppresses numbers through the neck's top-down path | scale each lateral, and the two routes of the stage-4 lateral separately | On the lab, removing the top-down route recovers 51 and loses 18, with 260 extra boxes; x0.5 recovers 19, loses 5. On val_dev the effect shrinks to +3 / 0 at x0.75 and reverses below x0.5. The lab effect was largely selection (the sample was picked for having misses). This became candidate T. |
| Amplifying a finer lateral recovers small text | in2, in3, in4 at x1.25 to x2 on val_dev | in4 x2: 8 recovered, 1 lost, 92 extra boxes (precision -6 points). About one box recovered per 12 added. Rejected. |
| Last-layer knob | bias shift 1-8, scale 0.5-4 at the fixed postprocess | Scale 0.5 with shift +3: +5 / 0 on val_dev. This became candidate K. T and K recover the same boxes and do not add. |

None of the 13 true blanks on val_dev is recovered by T or K. What they recover is one box that sat just under
box_thresh, plus geometry: a split merged, a vertical stack completed, loose-annotation mismatches.

## What I would do instead

- **If recall is the priority, do not ship `best_edited.pth` from part 1.** That edit trades recall for precision
  (test recall 0.890 -> 0.878). The original weights are the higher-recall checkpoint.
- **The free recall is in the postprocess, and it is small:** thresh 0.01 with box_thresh 0.2 gives recall 0.9533
  at precision 0.837 on val_dev (0.9484 / 0.836 at the fixed setting). Not checked on the holdout: the budget of
  five holdout evaluations is spent.
- **The blanks need training, not editing.** They are numbers the labels leave out in some images and mark in
  others, so the model learned to withhold them. Labelling unit numbers consistently (part 1 found that 39% of
  the "false positives" are legible unannotated numbers) and fine-tuning is the direct route, and it serves the
  goal of catching the address unit.
- **Vertical stacks** are the one recurring geometric failure (boxed digit by digit, or one digit only). More
  vertical examples in training, or merging vertically aligned boxes before recognition, would address it.
- **Small text is where recall drops** (train: 0.86 for boxes 16-24 px high at model resolution, 0.98 above 40 px).
  A larger test resize is the lever for that; it is fixed in this study and was not tested.

## Addendum: continued search (holdout cap lifted)

Five more routes, none delivered. Details in `notes.md` section 23.

| route | outcome |
|---|---|
| Are the extra boxes of the lateral amplifications withheld unit numbers? | No. Of 60 new boxes viewed for in4 x2: about 9 numbers (4 of them single digits inside annotated vertical stacks), 13 other lettering, 38 non-text. |
| Input contrast / brightness folded into the first layer | Contrast x1.25: 6 gained, 38 lost. Contrast x0.8 or brightness +0.3: 1-4 net gained for 130-150 extra boxes. |
| Channel-level scan of the four neck laterals (384 channels, gain 0 and 2) | Largest effect on missed text about +1 logit, with background raised; selective channels give +0.2. About +9 is needed. The withholding is distributed. |
| Last-layer family: scale s and contour offset c, `logit' = s (logit - L0 + c)` | On val_dev, s 0.3 with c 5 (R1) gave recall +0.7 point and precision +1.7. On the holdout: recall +0.1 [-0.6, +0.9], precision +2.6 [+1.5, +3.9]. |
| **R1 on the test set (one run, after a written freeze)** | **Failed.** Recall 0.8903 -> 0.8735 (3 gained, 24 lost), precision 0.317 -> 0.336 with every image counted. Worse than part 1's edit B on both counts. |

The test result changes one earlier statement: the small recall gains seen on val for the top-down edit, the
last-layer knob and the low-threshold postprocess setting (+0.4 to +0.7 point) should not be expected to carry to
the test distribution. Lowering the contour merged split digits on val and cost matches on test; I did not inspect
test images to confirm the cause.

Where this leaves the weight-editing route: every experiment agrees that the missed numbers are withheld by a
distributed decision formed in the deep backbone, and that nothing downstream can read them back out selectively.
The remaining levers are outside weight editing: consistent labels for unit numbers plus fine-tuning, vertical-stack
handling, and test resolution for small text.

## Not verified

- The test set was used once in part 2, for R1 only (see the addendum). Test recall (0.89) is lower than val (0.94)
  and its misses may be of a different mix; I did not look at them.
- The share "about 14 of 42 are label geometry" is my reading of the crops in `viz/fn_dev/`, not a measured quantity.
- The postprocess setting above is a val_dev number only.

## Files

`notes.md` sections 14-22 (numbers, decision rule, holdout); scripts `ws/80_*.py` to `ws/91_*.py` and `ws/fnlab.py`;
logs `logs/fn_diag_*.txt`, `frontier_dev.txt`, `fn_scan.txt`, `depth_probe.txt`, `e3_ridge.txt`, `e4_neck_dir.txt`,
`topdown.txt`, `in5_paths_*.txt`, `lateral_dev.txt`, `knob_recall_dev.txt`, `recall_cands_dev.txt`,
`recall_holdout.txt`; crop sheets `viz/fn_dev/`, `viz/fn_train3000/`. The rejected candidates are kept as
`cand_R_T0.75.pth` and `cand_R_K_s0.5_d3.pth`. `weights/best_accuracy.pth` is untouched
(md5 33db92a5910fa568724bfd1a278eeae1).

## Addendum 2: overlapping-crop inference and multi-scale fusion (2026-11-13, validation only)

Asked by the user. Original weights, forward only, fixed postprocess; details in `notes.md` section 24.

- **Tiled (overlapping-crop) inference does not help.** Images are already seen at about native resolution and the
  missed numbers are not small, so tiles only add resolution by upscaling, and upscaling floods the output with
  boxes (precision 0.76 -> 0.51 at 1.5x, 0.34 at 2x) without recall. Tiles at the same scale lose 2.6 pt of recall.
- **A smaller test resize helps precision.** Short side 640 instead of 736: precision +5.7 pt (dev) and +5.4 pt
  (holdout), recall +0.5 / +0.7 pt (inside the noise, not negative). Cheaper to run.
- **Multi-scale max-fusion (1.0x with 0.875x or 0.75x)** is the only recall-positive setting: +1.2 to +1.4 pt on dev,
  +0.3 to +0.4 pt on holdout (not significant there), for 1-2 pt of precision and a second forward pass.
- Not run on test.
