# ConvNeXt Detection Backbone Distillation — Design Sketch

Goal: replace the 27.8M-param DINOv3-ConvNeXt-tiny detection backbone with a
3–5M-param ConvNeXt-V2 femto/atto student that retains most of its accuracy on
POD text detection, cutting Metal latency from ~200 ms to ~40–60 ms.

Method: task-specialized feature distillation, adapted from EdgeCrafter (Liu et
al., arxiv 2603.18739). Two-stage: (1) feature-align an unlabeled-only
distillation pass on a large image pool, then (2) fine-tune the distilled
student with the existing labeled detection pipeline.

This document describes design only — no code yet.

---

## 1. Inputs you already have

| Asset | Path | Role |
|---|---|---|
| Task-adapted teacher | `weights/dinov3/convnext_det_unfreeze.pth` | Stage-1 already done — DINOv3-ConvNeXt-tiny fine-tuned on POD detection |
| Labeled POD detection set | (your existing det train split) | Stage-3 fine-tune |
| Unlabeled POD images | `images/` (any size, ~∞ samples) | Stage-2 distillation pool |

Stage 1 (task-adapt the teacher) is **already done** — that's what
`convnext_det_unfreeze.pth` is. You skip the entire teacher-prep step from the
paper.

---

## 2. Stage 2 — feature distillation (unlabeled)

### 2.1 Architecture

```
                                    ┌───────────────────────────┐
   raw image ──[same augment]──► T: │ DINOv3-ConvNeXt-tiny       │── F^T_3, F^T_4
                                    │ (frozen, eval mode)         │   teacher feats
                                    └───────────────────────────┘   (no_grad)
                                                                       │
                                    ┌───────────────────────────┐      │  MSE
   raw image ──[same augment]──► S: │ ConvNeXt-V2-femto (or atto)│      │
                                    │ (trainable)                 │── F^S_3, F^S_4 ──► φ_3(F^S_3), φ_4(F^S_4)
                                    └───────────────────────────┘
```

- **Teacher**: load `convnext_det_unfreeze.pth` into the existing
  `ConvNeXtDetBackbone`. Set `eval()`, `requires_grad_(False)`. Wrap forward in
  `torch.no_grad()`. Returns 4 feature maps `(F^T_1, F^T_2, F^T_3, F^T_4)` at
  strides `[4, 8, 16, 32]`.
- **Student**: new `ConvNeXtV2Backbone` wrapping
  `facebook/convnextv2-femto-1k-224` (or `-atto-1k-224`). Same 4-stage output
  interface. Init from HF ImageNet weights.
- **Adapter**: per-stage 1×1 conv `φ_s: C^S_s → C^T_s`, applied to student
  features before the loss. Two adapters total (stages 3 and 4 only).
- **What we align**: only stages 3 and 4 (strides 16 and 32 — the semantic
  features). EdgeCrafter found "last two blocks" is the sweet spot. Stages 1–2
  are not aligned because their features are dominated by low-level patterns
  that the student can learn from the data alone; aligning them over-constrains
  the student and hurts capacity for semantic features.

Channel dims for reference:

| Stage | Stride | Teacher (tiny) | Student (femto) | Student (atto) |
|---|---|---|---|---|
| 1 | 4  | 96  | 48  | 40 |
| 2 | 8  | 192 | 96  | 80 |
| **3** | **16** | **384** | **192** | **160** |
| **4** | **32** | **768** | **384** | **320** |

Adapter parameter cost (negligible):
- femto: 1×1×192×384 + 1×1×384×768 = ~370k params
- atto: 1×1×160×384 + 1×1×320×768 = ~308k params

### 2.2 Loss

Per-stage normalized MSE on adapted student vs. teacher features:

```
L_s = (1 / (B · C^T_s · H_s · W_s)) · || φ_s(F^S_s) − F^T_s ||²_2
L_distill = L_3 + L_4
```

Implementation notes:
- Normalize teacher features per-channel before MSE — divide by per-channel
  running mean+std (or just per-stage L2 normalize). Raw MSE on un-normalized
  features puts disproportionate weight on stages with larger magnitudes.
  EdgeCrafter omits this (their late-block features are LayerNormed already);
  for ConvNeXt the stage outputs are *not* LayerNormed, so add LN-then-MSE or
  L2-normalize both sides before MSE.
- Equal weights for the two stages — no need to tune λ per stage. If one stage
  dominates, normalize harder.
- Optional addition: cosine-similarity loss `L_cos = 1 − cos(φ(F^S), F^T)` as
  a second term with weight 0.5. Sometimes more stable than pure MSE when
  feature magnitudes drift.

### 2.3 Dataset prep

Single folder `images/` of unlabeled POD images. No annotations needed.

Augmentation (deliberately mild — we want clean teacher signal, not robustness):
- Resize short side to 1024, random crop to **640×640**.
- Random horizontal flip (p=0.5).
- Random color jitter (brightness/contrast 0.2, saturation 0.1, hue 0.02).
- **No mosaic, no cutmix, no random erasing** — these break the teacher's
  features the student is trying to imitate.

Normalization: same ImageNet mean/std as Stage-3 detection. **Teacher and
student must see the same pixels** — apply the augment once, feed both networks
the identical tensor.

Dataset class outline:

```python
class POdDistillDataset(Dataset):
    def __init__(self, images_dir, transform):
        self.paths = sorted(glob(f"{images_dir}/**/*.[jp][pn]g", recursive=True))
        self.transform = transform        # albumentations or torchvision

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"]   # one tensor, used twice
```

DataLoader: `num_workers=8`, `pin_memory=True`, `persistent_workers=True`,
`drop_last=True`.

**How many images?**

| Image count | Expected quality | Notes |
|---|---|---|
| 10k POD only | mediocre | won't generalize off-distribution |
| 50k POD only | good | minimum viable for production |
| **POD + ImageNet-1K (1.28M)** | **recommended** | matches the paper's recipe; teacher features remain in-distribution because ImageNet was DINOv3's pretraining domain |
| POD + LAION subset | best | overkill for POD-only deployment |

Recommended: mix POD images with ImageNet-1K at roughly 1:1 sampling. Use a
weighted sampler so each batch sees both.

### 2.4 Optimizer / schedule

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | standard for ConvNeXt |
| LR (peak) | **1e-3** for femto/atto | smaller models tolerate higher LR |
| Weight decay | 0.05 | matches ConvNeXt-V2 recipe |
| Betas | (0.9, 0.999) | default |
| LR schedule | cosine to 1e-6 | smooth decay |
| Warmup | 5 epochs linear from 1e-6 | prevents adapter blow-up |
| Total epochs | **100** if using ImageNet+POD; **300** if POD-only (~50k images) | match step count to roughly 1.5M iters at bs=128 |
| Batch size | 128 (one H100) / 64 (one A100/4090) / 32 (one M-series) | adapt to mem |
| Grad clip | 1.0 | safety |
| EMA | enabled, decay tied to schedule (see below) | stabilizes loss curves; use EMA weights for Stage 3 init |
| Precision | bf16 mixed | fp16 can underflow MSE on small feature magnitudes |
| Input resolution | **384×384** | see "Resolution choice" below |

#### Resolution choice

ConvNeXt is fully convolutional, so the student learns a function `pixels →
features` that is identical at any image size. Training resolution affects
speed × per-image signal density, not feature quality. DINOv3's own
distillation pipeline trains at 224, ConvNeXt-V2 paper trains at 224.

| Train res | FLOPs vs 640 | Per-image stage-32 samples | Notes |
|---|---|---|---|
| 640² | 1.00× | 20×20 = 400 | overkill; what the doc originally said |
| **384²** | **0.36×** | **12×12 = 144** | **recommended** — best speed/signal trade-off |
| 256² | 0.16× | 8×8 = 64 | fast; thin signal at deepest stage |
| 224² | 0.12× | 7×7 = 49 | DINOv3's choice; works but stage-32 MSE gets noisy |

Stage 3 fine-tunes at the full POD detection resolution (1280×704). Because
ConvNeXt is translation-equivariant, the resolution transfer takes ~20 epochs
of fine-tune to converge regardless of distillation resolution.

#### EMA decay choice

The EMA decay must match schedule length, otherwise the EMA either lags too
far behind (decay too high) or barely smooths anything (decay too low).

| Total steps | Decay |
|---|---|
| < 50k | 0.999 |
| **50k–500k** (typical POD-only run) | **0.9995** ← train script default |
| > 500k | 0.9999 |

EMA gives a smaller boost for pure-distillation loss (MSE is already smooth)
than for noisy classification/detection losses. Still worth keeping — costs
~20MB extra memory and one in-place update per step.

Wallclock budget estimate (femto student, POD-only ~50k images, bs=128, 100 ep,
**at 384²**):
- ~400 steps/epoch → 40k steps total
- ~0.08 s/step on H100 → ~55 min (~1 hr)
- One 4090: ~3× slower → ~3 hr
- One M3 Ultra (MPS): ~10× slower → ~10 hr

At 640² (the old recommendation) those times roughly 2.8×. The 384 default is
fast enough that a full run is feasible overnight on a 4090.

If on a tight budget, halve to 50 epochs — quality drops ~1-2 H-mean points
typically.

### 2.5 Training loop sketch

```python
teacher = build_teacher().eval().requires_grad_(False).cuda()
student = build_student().cuda()
adapters = nn.ModuleList([Conv2d(c_s, c_t, 1) for c_s, c_t in [(192,384),(384,768)]]).cuda()

optim = AdamW([*student.parameters(), *adapters.parameters()], lr=1e-3, weight_decay=0.05)
sched = CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-6)

for x in loader:                          # x: (B, 3, 640, 640), already normalized
    x = x.cuda(non_blocking=True)
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        t_feats = teacher(x)              # list of 4 maps
    with autocast(dtype=torch.bfloat16):
        s_feats = student(x)
        loss = 0
        for s_idx, t_idx in [(2, 2), (3, 3)]:      # stages 3 and 4 in 0-indexed
            t = layernorm_channel(t_feats[t_idx])
            s = layernorm_channel(adapters[s_idx-2](s_feats[s_idx]))
            loss = loss + F.mse_loss(s, t)
    loss.backward()
    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optim.step(); optim.zero_grad(); sched.step()
```

---

## 3. Stage 3 — detection fine-tune (labeled)

After Stage 2, the student backbone weights are dropped into a config that
mirrors `configs/det/PP-OCRv5/PP-OCRv5_convnext_det.yml`, swapping in the new
`ConvNeXtV2Backbone` with `in_channels` adjusted for femto/atto.

| Hyperparameter | Value | Notes |
|---|---|---|
| Initialization | Stage-2 EMA student weights (backbone), random for LKPAN/DBHead | discard adapters |
| LR (backbone) | 1e-4 | low — preserve distilled features |
| LR (LKPAN+head) | 5e-4 | higher — these are random |
| Schedule | cosine, warmup 1 epoch | |
| Epochs | 100 on labeled POD | matches existing recipe |
| Input shape | same as current pipeline (1280×704 or similar) | the distilled backbone is pure-conv, can do any res |
| Augmentation | full det augmentation (random crop, color jitter, etc.) | same as current ConvNeXt det config |

---

## 4. Metrics

### 4.1 Stage 2 training metrics (per step / per epoch)

- **MSE per stage** (`L_3`, `L_4`) — primary loss curve
- **Cosine similarity per stage** — `cos(φ(F^S), F^T)` averaged over tokens.
  Goes from ~0.0 (random init) to ~0.85–0.95 (well-distilled).
- **Adapter weight norm** — sanity check it isn't drifting unbounded
- **Grad norm** — should stabilize <1.0 after warmup

### 4.2 Stage 2 eval (proxy quality, every N epochs)

Run on a small POD val subset (~200 images):
- Per-stage cosine sim between teacher and student features → "distillation
  fidelity"
- Per-stage feature MSE (un-normalized) → easy to compare runs

Optional but informative — **mini fine-tune probe**: every 25 Stage-2 epochs,
fork the student, attach LKPAN+DBHead, train for 5 epochs on the labeled POD
train set, eval H-mean on val. Catches "loss is going down but features are
useless" failure modes.

### 4.3 Final metrics (after Stage 3)

POD detection val set:
- **H-Mean** (target: within 2 points of teacher's H-Mean)
- **Precision / Recall** at the operating threshold
- **AP@[0.5:0.95]** if you track it

Latency on Metal (MNN, fp16, 1280×704):
- Teacher (current): 200 ms
- Femto student target: **45–60 ms** (~3.5–4× speedup)
- Atto student target: **35–45 ms** (~4.5–5.5× speedup)

Model size:
- Teacher: 28M params / ~110 MB fp32 / ~55 MB fp16
- Femto: ~5M / ~20 MB / ~10 MB
- Atto: ~4M / ~15 MB / ~7.5 MB

### 4.4 Acceptance criteria

Ship the femto student if:
1. H-Mean within **−2 absolute points** of teacher on POD val
2. Metal p95 latency **< 80 ms**
3. No regression on a manually-curated 50-image "hard" set (small text,
   skew, low contrast)

Otherwise: re-train Stage 2 with more data / longer schedule, or fall back to a
nano-sized student (~16M params, ~80–100 ms).

---

## 5. Risks & open questions

1. **ConvNeXt-V2 femto/atto weights are ImageNet-1K only** — much weaker
   initialization than DINOv3. Stage-2 distillation has to do more work to
   close the gap than EdgeCrafter's ViT case (their student inits from
   scratch+stem). Mitigation: longer Stage-2 schedule, larger image pool.

2. **No DINOv3 self-supervised pretraining for the student.** Considered:
   running DINOv3-style SSL pretrain on POD before distillation. Probably
   not worth the complexity — feature distillation from a task-adapted teacher
   subsumes most of the SSL benefit per the EdgeCrafter ablations.

3. **Channel-dim mismatch is large** (192→384, 384→768 for femto; 160→384,
   320→768 for atto). The adapter has to learn a 2× expansion. Should still
   work — 1×1 conv is generic — but watch the early-epoch loss curve. If it
   plateaus high, try a tiny 2-layer MLP adapter instead.

4. **Output-stride mismatch**: ConvNeXt-V2 base config uses stride-4 stem,
   matching teacher. Double-check the HF femto/atto configs use the same.

5. **Resolution mismatch**: teacher was fine-tuned at the POD resolution
   (1280×704), but distillation runs at 384×384. ConvNeXt is fully
   convolutional so this is fine — features are translation-equivariant and
   stride-correct at any resolution. Stage 3 will fine-tune at full res.

---

## 6. Implementation order (when ready)

1. Add `ConvNeXtV2Backbone` class to `torchocr/modeling/backbones/`, mirroring
   `ConvNeXtDetBackbone`'s interface (4-stage tuple output) but using
   `facebook/convnextv2-femto-1k-224` from HF.
2. Build `tools/distill/` with: dataset, train loop, adapter module, config.
   Self-contained — don't reuse `BaseModel` plumbing (it's task-specific).
3. Add a Stage-3 config `configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_det.yml`
   pointing to the distilled backbone weights.
4. Bench script for the final exported MNN model — reuse the
   `bench_convnext_pod.py` pattern from the ONNX-optimization investigation.

Estimated dev time: 2–3 days for plumbing, then training wallclock dominates.
