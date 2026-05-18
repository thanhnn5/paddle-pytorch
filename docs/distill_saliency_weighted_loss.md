# Plan: Saliency-Weighted Distillation Loss (Option B)

Status: **planned, not yet implemented.** Refactor of `tools/distill/` into
decoupled modules precedes implementation so this change touches only `loss.py`
and `model.py`.

## Why

In POD images the address (the *only* text region we care about for detection)
typically occupies 5–15% of the pixels. Under the current pipeline:

- `RandomResizedCrop(scale=(0.5, 1.0))` means ~30% of crops contain no address
  at all, and the rest still have 85–95% background.
- The feature-distillation MSE is averaged over every spatial position
  equally. The student can drive total loss down by matching the teacher on
  background (easy: backgrounds are uniform, low-magnitude features) while
  doing a mediocre job on text regions (hard: distinctive, high-magnitude
  features).
- Gradient is therefore dominated by background, and the very features that
  matter for Stage-3 detection are *under-trained*.

The FGD paper (Feature Knowledge Distillation for Detection) reports 1–3 mAP
losses on COCO when ablating foreground masking from feature distillation.
For our task (very small foreground / image) the effect is likely larger.

We can't use labels — Stage 2 is unlabeled by design. But the teacher itself
emits a per-pixel text probability map: that's the entire output of DBHead.
We can use it as a free saliency mask.

## Design

### Pipeline change

Currently `build_teacher()` loads only `ConvNeXtDetBackbone`. Change it to
load the **full BaseModel** (backbone + LKPAN + DBHead). Two outputs per
forward:

```
teacher(x) -> (features, prob_map)
  features:  list[Tensor]  per-stage feature maps as today
  prob_map:  Tensor (B, 1, H_out, W_out)  sigmoid of DBHead, in [0, 1]
```

`H_out, W_out` = input resolution. For stage-3 alignment (stride 16) we
downsample with bilinear interp; same for stage-4 (stride 32).

### Loss formula

For each aligned stage `s` with feature spatial size `(H_s, W_s)`:

```
saliency_s = F.interpolate(prob_map, size=(H_s, W_s), mode='bilinear', align_corners=False)
weight_s   = α + β * saliency_s                # shape (B, 1, H_s, W_s)
diff_s     = (φ_s(F^S_s) - F^T_s) ** 2         # shape (B, C^T_s, H_s, W_s)
L_s        = (diff_s * weight_s).sum() / (weight_s.sum() * C^T_s)
L_distill  = mean_s(L_s)
```

- `α = --saliency-alpha` (default 0.1) — minimum weight for background. Keeps
  the student learning *something* on background; setting α=0 would let
  per-pixel weight go to 0 on confident background and stop background
  learning entirely.
- `β = --saliency-beta` (default 1.0) — additional weight on text regions.
  With α=0.1, β=1.0, text regions weigh ~11× background.
- Normalization by `weight_s.sum()` (not by num pixels) ensures the overall
  loss magnitude is invariant to image content. Otherwise images with lots
  of text would dominate.

LayerNorm on features happens before the squared diff, as today.

### Reusing the teacher's saliency for the cosine loss

Cosine similarity loss currently averages across all spatial positions
uniformly. With saliency available, weight it too:

```
cos_s = (cos_per_pixel * saliency_s).sum() / saliency_s.sum()
```

This makes both the MSE and cosine terms text-region-focused. Or leave cosine
unweighted — debatable; let it be a flag.

### CLI additions

```
--saliency-weight       enable saliency weighting (default off; off = current behavior)
--saliency-alpha        background weight floor (default 0.1)
--saliency-beta         foreground weight multiplier (default 1.0)
--saliency-from         "head" | "stage" | "off"
                        head  = DBHead sigmoid (preferred — task-aligned)
                        stage = L2-norm of teacher's deepest feature map
                                (fallback if loading full BaseModel is awkward)
                        off   = no saliency, current behavior
--saliency-detach       detach saliency before use (default true; otherwise
                        backprop tries to update the frozen teacher)
```

Default off keeps backward compatibility with current runs and sweep results.

## Implementation steps

Touched files (in order):

1. **`tools/distill/model.py`** (new in refactor): replace `build_teacher()`
   with `build_teacher_full()` returning a wrapper module that exposes both
   features and the per-pixel prob map.
2. **`tools/distill/loss.py`** (new in refactor): add
   `feature_distill_loss(s_feats, t_feats, adapters, align, *, saliency=None,
   alpha=0.1, beta=1.0, layernorm=True, cosine_weight=0.5)`.
3. **`tools/distill/train.py`**: pass saliency through to the loss.
4. **`docs/convnext_distillation.md`**: add a "Saliency weighting" subsection
   referencing this doc.

Expected diff: ~80 lines net.

## What to verify

Once implemented, run a small A/B sweep:

```bash
# Baseline: no saliency
LRS="1e-3" IMG_SIZES="384" COSINE_WEIGHTS="0.5" \
  ALIGN_STAGES="2 3" SEEDS="0 1 2" SWEEP_TAG=no_saliency \
  ./scripts/distill_sweep.sh

# With saliency
LRS="1e-3" IMG_SIZES="384" COSINE_WEIGHTS="0.5" \
  ALIGN_STAGES="2 3" SEEDS="0 1 2" SWEEP_TAG=saliency \
  EXTRA_ARGS="--saliency-weight --saliency-alpha 0.1 --saliency-beta 1.0" \
  ./scripts/distill_sweep.sh

python tools/distill/compare.py \
  'output/sweeps/no_saliency/*/summary.json' \
  'output/sweeps/saliency/*/summary.json' \
  --group-seeds --sort-by cos_last
```

Expected: saliency-weighted runs have ~5–15% lower `mse_ln_last` on stage 4
and ~0.01–0.05 higher `cos_last`. Bigger if probe-set images are address-rich.

The real test is Stage 3 detection AP, but that takes hours to measure.
Probe metric is a fast proxy and what the sweep will report.

## Failure modes to watch for

1. **DBHead output is noisy at the start of training (teacher checkpoint
   accuracy varies on out-of-distribution crops).** Mitigate by clipping
   saliency to `[saliency_clip_low, 1.0]` or by detaching after a small
   power transform (`saliency = saliency.pow(0.5)`) to soften extreme
   confidence.
2. **Tiny α can cause loss spikes on all-background crops** (when no pixel
   has high saliency, the loss becomes a high-variance average over a few
   weighted pixels). Keep α ≥ 0.05.
3. **Teacher prob map is at full resolution.** For a 384² input the prob
   map is 384×384 — bilinear-down to the 12×12 stage-4 feature map. This is
   fine but be aware that fine-grained text outlines get averaged out at
   stride-32 resolution. The supervision signal at deep stages is
   coarse-grained "is there text in this 32×32 patch" rather than "is this
   pixel text".

## Open questions

- Should saliency be drawn from the *teacher's* prediction or from the union
  of teacher + student predictions (curriculum-style: as student improves,
  trust its high-confidence predictions too)? Probably teacher-only; the
  student is the thing being trained.
- Should we additionally mask the per-pixel loss to *only* high-saliency
  regions (hard mask) rather than soft-weight? Hard mask is simpler but
  loses gradient on the borderline pixels; soft weight is more standard.
- For Stage 3 fine-tune, does saliency-weighted distillation transfer better
  than uniform? Only an end-to-end H-mean comparison answers this.
