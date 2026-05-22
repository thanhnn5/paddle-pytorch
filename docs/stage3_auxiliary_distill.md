# Plan: Stage-3 Auxiliary Distillation Loss

Status: **planned, not yet implemented.** Inspired by RT-DETRv4 (arxiv
2510.25257), adapted for our ConvNeXt detection pipeline.

## Motivation

Stage-2 feature distillation (`tools/distill/`) produces a femto student
backbone whose features approximate the teacher's on a mixed image pool.
Stage-3 then fine-tunes the full student detector (backbone + LKPAN + DBHead)
on labeled POD data.

The problem we observed:

> Stage-3 detection fine-tune lets the student backbone drift away from the
> Stage-2-learned features as it specializes for the POD labels. The student
> achieves good val performance (H≈0.92) but generalizes poorly on test
> (recall 0.89 → 0.64). The backbone's features collapsed to the narrow POD
> appearance manifold.

The teacher's features are *more general* — both because the teacher itself
was task-adapted on a broader-distribution detection set, and because Stage 2
exposed the student to teacher features on a wider image mix (COCO + ICDAR).
But during Stage 3, no teacher signal is present, so the student's backbone is
free to overfit.

**Auxiliary distillation in Stage 3** adds a teacher-feature alignment term
alongside the detection loss. The student is still optimized for detection,
but is *anchored* to the teacher's feature distribution as a regularizer.

RT-DETRv4 demonstrates this pattern (see local repo
`/Users/thanhnn5/Workspace/Jitsu/POD/RT-DETRv4`): they train detection
end-to-end with auxiliary cosine-similarity distillation to off-the-shelf
DINOv3 ViT-B features, achieving SOTA real-time detection on COCO. They call
this "Dense Spatial Imitation (DSI)" loss and auto-tune its weight with
"Gradient-guided Adaptive Modulation (GAM)" — we adopt the loss but
hand-tune the weight, skipping GAM as overkill for one model config.

## Reference: how RT-DETRv4 does it

| Element | RT-DETRv4 | Notes for our case |
|---|---|---|
| Teacher | Off-the-shelf DINOv3 ViT-B/16 (LVD-1689M) | We reuse our task-adapted ConvNeXt teacher — sharper signal for OCR |
| When | Joint with detection (no separate stage) | We add as auxiliary to Stage 3 only |
| Where | Encoder F5 output (post-DETR-encoder) | We align backbone stage 3 & 4 outputs (matches Stage 2) |
| Loss | `(1 - cos_sim).mean()` on L2-normalized flat tokens | Same |
| Adapter | 1× linear `hidden_dim → 768` | 1×1 conv per stage to match teacher channels |
| Weight | Static `5–20` + GAM auto-tune | Static, manually tuned in {1, 5, 10} |

The differences from our Stage 2 (deliberate):
- **No saliency weighting.** RT-DETRv4 doesn't bother and it works for them.
  Stage 3 has the actual detection loss to anchor "what matters" — no need
  for an additional signal-region focus.
- **Cosine-only loss**, not MSE+cos hybrid. Simpler, slightly cheaper.
- **Backbone stage features**, not LKPAN output. Aligning closer to the raw
  representation keeps the LKPAN free to specialize for detection.

## Design

### Data flow per training step

```
batch (images, targets)
  │
  ├── student.forward(images)
  │     ├── backbone(images)  → [stage1, stage2, stage3, stage4]
  │     │     └── [forward hook]  captures into trainer._backbone_feats
  │     ├── neck(backbone_out)
  │     └── head(neck_out)  → student outputs
  │
  ├── det_loss = criterion(student outputs, targets)     (existing)
  │
  ├── if cfg.AuxDistill.enabled:
  │     ├── with torch.no_grad():
  │     │       teacher_feats, _ = teacher(images)
  │     │       # teacher_feats: list[Tensor] from TeacherBackboneOnly
  │     ├── student_feats = trainer._backbone_feats
  │     ├── for stage in align_stages:
  │     │       s_proj = adapter[stage](student_feats[stage])
  │     │       loss_s = (1 - cosine_sim(L2norm(s_proj), L2norm(teacher_feats[stage]))).mean()
  │     └── distill_loss = mean(loss_s for s in align_stages)
  │
  └── total = sum(det_loss.values()) + cfg.AuxDistill.weight * distill_loss
  
  backward
  optimizer.step()   # includes adapter params via model.aux_distill.parameters()
```

### Loss formula

For each aligned stage `s` with student features `F^S_s` (shape `B×C^S×H×W`)
and teacher features `F^T_s` (shape `B×C^T×H×W`):

```
F^S_proj = adapter_s(F^S_s)                              # 1x1 conv: C^S -> C^T
F^S_flat = F^S_proj.flatten(2).permute(0, 2, 1)          # B x N x C^T
F^T_flat = F^T_s.flatten(2).permute(0, 2, 1)             # B x N x C^T

L_s = (1 - cos_sim(L2norm(F^S_flat), L2norm(F^T_flat), dim=-1)).mean()
```

Total auxiliary loss = `mean(L_s for s in align_stages)`.

Notes:
- Teacher features are bilinear-resized to match student spatial size if they
  differ (they shouldn't if both are ConvNeXt at the same input resolution,
  but safety net).
- Cosine sim is per-token (per-spatial-location), averaged across all tokens
  and the batch.
- Adapter has no bias on the proj — RT-DETRv4 omits bias and we follow suit.
- L2 normalize before cos_sim (technically redundant since `F.cosine_similarity`
  normalizes internally, but explicit makes the formula clearer).

### Config schema

New top-level block in detection YAMLs:

```yaml
AuxDistill:
  enabled: false                # default: off (preserves current behavior)
  teacher_ckpt: weights/dinov3/convnext_det_unfreeze.pth
  align_stages: [2, 3]          # 0-indexed backbone stage indices
  weight: 5.0                   # multiplier on the distillation term
  # Teacher architecture is hardcoded to ConvNeXtDetBackbone matching our
  # current teacher. If we need to swap teachers later, add `teacher_arch`
  # and `teacher_config` fields.
```

Default off. Enable via `-o AuxDistill.enabled=true` in the Phase-2
invocation of `finetune_femto_mobile.sh` (and equivalents).

### Phase 1 vs Phase 2

`scripts/finetune_femto_mobile.sh` runs in two phases:
- **Phase 1** (~25 epochs): backbone FROZEN. Random LKPAN+DBHead stabilizes
  against fixed Stage-2 features.
- **Phase 2** (~50 epochs): backbone UNFROZEN. Joint fine-tune.

**Auxiliary distillation runs in Phase 2 only.** In Phase 1 the backbone has
no gradient, so the distill loss can't change backbone features anyway — it
would only update the adapter, which is pointless before Phase 2 turns on the
backbone. The adapter is initialized fresh at Phase-2 start.

Implementation: enable via Phase-2 `-o` override; Phase-1 leaves
`AuxDistill.enabled=false` (the YAML default).

## File-by-file changes

### 1. `torchocr/losses/aux_distill.py` (new, ~60 LOC)

```python
class AuxDistillLoss(nn.Module):
    """Per-stage cosine-similarity distillation loss with channel adapters.

    Owns the 1x1 conv adapter per aligned stage. Designed to be registered as
    a submodule of BaseModel so its parameters are picked up by the optimizer.
    """
    def __init__(self, student_channels: list[int], teacher_channels: list[int],
                 align_stages: list[int], weight: float = 5.0):
        super().__init__()
        self.align_stages = align_stages
        self.weight = weight
        self.adapters = nn.ModuleList([
            nn.Conv2d(student_channels[s], teacher_channels[s], 1, bias=False)
            for s in align_stages
        ])

    def forward(self, student_feats: list, teacher_feats: list) -> Tensor:
        # ... per-stage cos-sim, return scalar
```

Self-contained; can be unit-tested with random tensors.

### 2. `torchocr/engine/trainer.py` (modified, ~70 LOC added)

- **`__init__`**: read `cfg["AuxDistill"]`. If enabled:
  1. Build teacher via `tools.distill.model.build_teacher(ckpt, device)`
     (reuses Stage-2 infrastructure → `TeacherBackboneOnly` wrapper whose
     forward returns `(features, None)`).
  2. Instantiate `AuxDistillLoss(student_channels, teacher_channels,
     align_stages, weight)` and attach as `self.model.aux_distill` so its
     parameters flow into the optimizer naturally.
  3. Register a forward hook on `self.model.backbone` that stashes its
     output list into `self._backbone_feats`.

- **Training step**: after computing `det_loss`, if aux_distill is enabled:
  ```python
  with torch.no_grad():
      teacher_feats, _ = self.teacher(images)
  student_feats = self._backbone_feats
  loss_distill = self.model.aux_distill(student_feats, teacher_feats)
  loss = sum(det_loss.values()) + self.model.aux_distill.weight * loss_distill
  ```
  Track `loss_distill` separately for logging.

- **Checkpoint save/load**: aux_distill adapter weights are saved as part of
  `model.state_dict()` automatically (because it's a submodule). No special
  handling needed. They are *not* loaded into a non-AuxDistill model later —
  state_dict load with `strict=False` drops them silently. Acceptable.

### 3. `torchocr/optimizer/__init__.py` (modified, ~5 LOC)

The current param-group builder splits params by name prefix
(`backbone.*`, `neck.*`, `head.*`). Add a fourth group for `aux_distill.*`
with `aux_distill_lr_mult: 1.0` default (same as head). Negligible change.

### 4. Detection YAML configs (~10 lines each)

Add the `AuxDistill` block (with `enabled: false`) to:
- `configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_mobile_det.yml`
- `configs/det/PP-OCRv5/PP-OCRv5_convnextv2_femto_balanced_det.yml`
- `configs/det/PP-OCRv5/PP-OCRv5_convnextv2_pico_balanced_det.yml`

### 5. `scripts/finetune_femto_mobile.sh` (~3 lines added in Phase 2 block)

Add to Phase 2 `-o` overrides:
```
AuxDistill.enabled=true \
AuxDistill.teacher_ckpt=weights/dinov3/convnext_det_unfreeze.pth \
AuxDistill.weight=5.0 \
```

Phase 1 untouched.

## Hyperparameter choices

| Parameter | Default | Rationale |
|---|---|---|
| `weight` | 5.0 | RT-DETRv4 uses 5 for S/M and 15–20 for L/X. Our femto is closer to S → start at 5. Sweep {1, 5, 10} after first run. |
| `align_stages` | `[2, 3]` | Strides 16 and 32 — same as Stage 2. Deep stages carry the most semantic content. Aligning stage 0 / 1 would over-constrain low-level features. |
| Loss type | Cosine sim | RT-DETRv4 style. Simpler than MSE+cos, well-understood. |
| Saliency weighting | OFF | User decision — Stage 3 has detection labels to anchor where signal matters. |
| Phase 1 enable | OFF | Backbone frozen → distill loss can only update adapters → pointless before Phase 2. |
| Phase 2 enable | ON | The actual use case. |
| Adapter LR multiplier | 1.0 (= head LR) | Adapters are freshly initialized; same logic as head. |

## Compute & memory

- **Wallclock**: teacher (~28M params, ConvNeXt-tiny) forward on every batch.
  Roughly +40–50% per-step time vs current Phase 2. On a 4090: Phase 2 goes
  from ~3 hr → ~5 hr for 50 epochs. On a single A100: ~3 hr.
- **Memory**: teacher params on GPU (~110 MB fp32 / ~55 MB fp16). Activations
  not retained (no_grad). Roughly doubles the model parameter footprint;
  inconsequential at our scales.
- **No grad** on the teacher — `requires_grad_(False)` + `torch.no_grad()` —
  so backprop graph is the same size as a non-distill run.

## Expected effect on metrics

Hypothesis (extrapolated from RT-DETRv4 results and our val/test gap):

| Metric | Current Phase-2 (no aux distill) | With aux distill (predicted) |
|---|---|---|
| Val H-mean | ~0.92 (unchanged or slightly lower — see below) | ~0.91–0.92 |
| Test H-mean | ~0.78–0.83 (extrapolating from femto-balanced+sal+mix) | ~0.85–0.87 |
| Test recall | ~0.64 (the failure mode) | **~0.80–0.85** ← the target improvement |
| Test precision | ~0.85 | ~0.85 (probably similar — distill regularizes, not specializes) |

The val H-mean *might* drop slightly with aux distill because we're partially
constraining the backbone from over-specializing to POD val patterns. The
expected payoff is on test, where the regularization prevents the
val-to-test feature collapse.

If the val H-mean drops by more than 1 point, the distillation weight is
too high — turn down to 1 or 2.

## Failure modes to watch for

1. **`weight` too high (e.g. >10)**: distill loss dominates, detection loss
   under-optimized. Symptom: detection H-mean drops on both val and test.
   Mitigation: lower `weight`, or set up a sweep over {1, 2, 5}.
2. **Teacher input mismatch**: Phase 2 uses detection augmentation (mosaic,
   copy-paste, 640² random crop). Teacher sees the same augmented input.
   This is FINE — teacher's features under mosaic'd inputs are still useful
   targets — but its features may be slightly worse than what it'd emit on
   clean images. We accept this; alternative is reprocessing each image
   twice with different aug, doubling DataLoader cost. Not worth it.
3. **Adapter collapse**: adapters might learn `proj(x) ≈ 0` if the cosine
   loss has a degenerate solution. Cosine sim with zero vectors is undefined
   → check for `nan`/`inf` in `loss_distill` during smoke tests. If observed,
   add an L2-norm-clamp before the cos_sim or switch to MSE on LN-normalized
   features (the Stage-2 formulation).
4. **Phase-1 to Phase-2 adapter initialization**: adapter is freshly created
   at Phase-2 start. Its outputs initially have near-zero cos_sim with
   teacher → loss starts at ~1.0 and drops. This is normal — should
   stabilize at 0.3–0.5 after a few hundred steps. Watch the first epoch's
   `loss_distill` trajectory.

## Test plan

1. **Unit test** (`tests/test_aux_distill.py`, new) — `AuxDistillLoss` with
   random tensors:
   - Forward returns scalar
   - `loss.backward()` populates adapter param `.grad`
   - Sanity: `loss ≈ 1.0` for orthogonal student/teacher, `loss ≈ 0.0` when
     student adapter is identity to teacher.
2. **Trainer integration smoke** — `python tools/train.py -c <femto_mobile.yml>
   -o AuxDistill.enabled=true Global.epoch_num=1 Train.loader.batch_size_per_card=2`
   on MPS. Verify:
   - Teacher loads cleanly
   - Backbone forward hook fires
   - `loss_distill` appears in the printed loss dict
   - One backward + optimizer step succeeds
3. **First-step loss sanity** — `loss_distill` ≈ 0.7–1.0 at step 0
   (adapter random init). Decreasing trend over 50 steps.
4. **End-to-end Phase-2 dry run** — `--max-steps 200` with the real
   `finetune_femto_mobile.sh` Phase-2 invocation. Verify training metrics
   look healthy.
5. **Disabled-by-default verification** — run a normal training without
   `AuxDistill.enabled=true` and confirm no behavioral change vs current
   master (numerical equivalence with a fixed seed).

## Implementation order

1. Write `torchocr/losses/aux_distill.py` (~60 LOC)
2. Unit test (test plan step 1) — pass before proceeding
3. Wire into `torchocr/engine/trainer.py` (teacher loading, hook, loss term)
4. Optimizer param-group plumbing (~5 lines in `torchocr/optimizer/__init__.py`)
5. Add `AuxDistill` block to femto-mobile YAML (default disabled)
6. Smoke tests (steps 2 & 3 of the test plan)
7. Update `finetune_femto_mobile.sh` Phase 2 with `-o AuxDistill.enabled=true ...`
8. Smoke test step 4 (end-to-end short run)
9. Add `AuxDistill` block to femto-balanced and pico-balanced YAMLs
10. Verify disabled-by-default still works (test plan step 5)
11. Commit + push

Estimated effort: **~2 hours coding + ~30 min smoke tests** = half a day.
Net change ~150 LOC.

## Open questions before starting

1. **Teacher arch hardcoded** to `ConvNeXtDetBackbone`. Fine for now. Add a
   `teacher_arch` field later if we ever need to swap teachers in Stage 3.
2. **Default `weight: 5.0`** — sweep {1, 5, 10} after the first end-to-end
   run lands? Or commit to 5 and only sweep if val H-mean drops noticeably?
   My take: commit to 5; sweep only if results disappoint.
3. **No CLI override of `align_stages`** — YAML only. Same convention as
   `Architecture.Backbone.size`. Fine.

## What we are NOT building (for now)

- **GAM (auto-tune of `weight`)** — implementation cost ~100 LOC, mostly
  worth it across many model sizes. For one student config (femto/pico),
  manual sweep is cheaper.
- **Linear-decay weight schedule** — could later replace static weight
  with `weight = w_start * max(0, 1 - epoch/total_epochs)`. Trivial to add
  if results need it; out of scope for v1.
- **Aux distill in Phase 1** — pointless since backbone is frozen.
- **Saliency weighting on the aux loss** — user opted out for simplicity.

## Cross-references

- Stage 2 distillation design: `docs/convnext_distillation.md`
- Stage 2 saliency-weighted loss: `docs/distill_saliency_weighted_loss.md`
- RT-DETRv4 reference: `/Users/thanhnn5/Workspace/Jitsu/POD/RT-DETRv4`
- Key RT-DETRv4 files:
  - `engine/rtv4/dinov3_teacher.py` — frozen teacher wrapper
  - `engine/rtv4/rtv4_criterion.py:76-115` — cosine-sim distillation loss
  - `engine/solver/det_engine.py:62-78` — teacher forward in training step
  - `engine/solver/det_solver.py:99-141` — GAM adaptive weight tuning
    (referenced but not adopted)
