# Experiment roadmap (from the 2026-11-10 survey)

Purpose: an ordered backlog to run when GPU capacity is available. Evidence and
sources are in [README.md](README.md) and `notes/`. Expected gains are
extrapolated from papers, not measured on our models.

Conventions:
- **Cost:** S = config or <50 lines; M = new module or data pass; L = new
  pipeline.
- **Evidence:** strong = peer-reviewed ablation; moderate = toolkit ablation;
  weak = single report or our own reasoning.
- Every experiment is one change against a fixed baseline, evaluated on the POD
  detection and recognition sets already used in `docs/eval_det_extended.md`.

## Phase 0 — checks that need no training

These can run on a laptop and gate later choices.

| ID | Check | Why | Done when |
|---|---|---|---|
| P0.1 | Trace how the ConvNeXt-V2 student was first initialized (`facebook/convnextv2-*` ImageNet weights or random) | Those weights are CC-BY-NC | Provenance written down; legal decision requested if they were used |
| P0.2 | Legal review of the DINOv3 licence for shipping a distilled student | Custom Meta licence with use restrictions | Written answer |
| P0.3 | Compare fp32 vs fp16 detector outputs on ~20 POD images; log the max sum of squares inside each GRN | GRN may overflow fp16 on large maps | Max value known; map difference quantified |
| P0.4 | Xcode Core ML performance report for the femto/pico detector | LayerNorm, GELU, GRN may fall back off the Neural Engine | Per-op compute unit listed |
| P0.5 | Count backbone / neck / head parameters of the exported recognizer | Head is estimated at ≥50% with 18,383 classes | Real split known |
| P0.6 | Histogram of text-line aspect ratios and character set in POD labels | Drives width buckets and dictionary pruning | Bucket widths and a POD charset proposed |

If P0.3 or P0.4 fails badly, D5 (a BatchNorm re-param student) moves ahead of
the other detection work.

## Phase 1 — detection (DB student distilled from the DINOv3-ConvNeXt teacher)

| ID | Change | Evidence | Cost | Expected |
|---|---|---|---|---|
| D1 | Align one deep fused feature (neck output or stride-32) instead of backbone stages 2–3 in `aux_distill` | Strong (RT-DETRv4), on COCO | S | Unknown sign on text; cheapest ablation |
| D2 | Foreground-weighted feature distillation: weight by the dilated text mask plus a small background weight | Strong on COCO (FGD), weak on text | S | +0.3 to +1.0 Hmean |
| D3 | Output-map distillation: student shrink/binary maps against the teacher's thresholded, dilated shrink map (`DistillationDilaDBLoss` exists) | Moderate (PP-OCRv2–v4 CML) | S | Up to a few Hmean in PP-OCRv2 |
| D4 | PP-OCRv6 loss on the student: Dice + Focal (γ=2.5) and training-only auxiliary heads | Moderate (PP-OCRv6 paper, in-house set) | S (code is ported) | Not isolated in the paper |
| D5 | Same teacher, BatchNorm re-param student (LCNetV4 small, RepViT or MobileOne class) at matched latency | Weak for accuracy; motivated by P0.3/P0.4 | M | Latency and fp16 robustness |
| D6 | Dynamic shrink ratio 0.4 → 0.6 in label generation; re-tune unclip ratio | Moderate (PP-OCRv4, +1.27) | S | ~+1 Hmean |
| D7 | Binarize and dilate inside the exported graph; keep only contour extraction on CPU | Moderate (FAST) | M | Post-processing latency |
| D8 | PP-OCRv6 tiny/small fine-tuned on POD as an external baseline | — | S | Reference point |

Order: D8 and D1 first (baseline and cheapest ablation), then D2 + D3 together
if each helps alone, then D4 and D6. Run D5 only if Phase 0 says so.

Fixed controls for all of Phase 1: same teacher checkpoint, same unlabeled
pool, same input size, same post-processing thresholds unless the experiment
changes them.

## Phase 2 — recognition (PP-OCRv5-mobile-style CTC model)

| ID | Change | Evidence | Cost | Expected |
|---|---|---|---|---|
| R1 | Restrict the dictionary to the POD charset and re-fit the head | Moderate (PP-OCRv6 tiny drops scripts) | S | Size and latency; accuracy neutral or better |
| R2 | DCTC self-distillation term on the CTC head | Strong (AAAI 2024) | S | +0.5 to +1.1 on SVTR-class models |
| R3 | Disable label smoothing on the NRTR guide loss when distilling | Moderate (PP-OCRv4 docs, size not reported) | S | Small |
| R4 | Width buckets at inference (e.g. 48×160 / 320 / 640) instead of one squashed width | Strong (SVTRv2 MSR, long-text results) | S–M | Long lines such as tracking numbers |
| R5 | Replace the NRTR guide with an SVTRv2-style semantic guidance module | Strong on 20M models, untested on small convnets | M | Largest upside, highest risk |
| R6 | Distil from a large same-dictionary teacher (PP-OCRv5 server, or SVTRv2-B retrained on our dictionary) using the existing CTC-logit distillation | Moderate (PP-OCRv4, v6) | M | ~+1 |
| R7 | CR-CTC two-view consistency | Strong in speech, none in OCR | M | Unknown; doubles forward cost |
| R8 | Temperature-scale CTC confidences on a held-out POD set | Weak | S | Better thresholds for fallback routing |

Order: R1 → R2 and R3 → R4 → R6 → R5 → R7. R8 can run any time.

Do not spend time on: EnCTC, Focal-CTC as a default, or frame-level KL between
CTC models with different architectures. All three hurt in DCTC's
re-implementation.

## Phase 3 — data and teachers (largest expected gain, most work)

| ID | Step | Evidence | Cost |
|---|---|---|---|
| T1 | Pseudo-label unlabeled POD crops with two independent teachers (a server recognizer and a VLM); keep exact agreements, queue disagreements for review | Moderate (PP-OCRv3 UIM +1.0, PP-OCRv5 practice) | M |
| T2 | Detection pseudo-boxes from the task-adapted teacher plus one server detector; keep agreeing boxes | Moderate | M |
| T3 | Two-model confidence filter on the training set (drop >0.95 from a weak model, <0.15 from a strong one) | Moderate (PP-OCRv4 DF +1.2) | S |
| T4 | One self-training round: relabel with the improved student plus teachers, retrain | Strong (Baek 2021, CRNN +2.8) | M |
| T5 | Rotation-pretext or DiG/SSM-style pre-training of the recognizer backbone on unlabeled crops | Strong for ViT, weak for small convnets (+0.6 in PP-OCRv3) | M–L |
| T6 | Targeted synthetic lines for rare POD strings (tracking numbers, postcodes) | Weak for our domain | M |
| T7 | Server VLM fallback for low-confidence labels, after deterministic field checks (check digits, formats) | Weak (design reasoning) | L |

Order: T3 → T1 + T2 → T4. T5 is worth running only if the labeled set stays
small. T6 only if T1 shows the teachers are unreliable on those strings.

Teacher outputs should be computed once and stored, so student training fits a
modest GPU.

## Phase 4 — deployment

| ID | Step | Depends on |
|---|---|---|
| X1 | Enumerated fixed shapes: 2–4 detector aspect buckets, 2–3 recognizer widths | P0.6, R4 |
| X2 | fp16 everywhere first; then 6–8-bit weight palettization on iOS | P0.3 |
| X3 | int8 only where a runtime needs it; keep DB head outputs and the CTC classifier at 16-bit or fp16; use QAT if PTQ loses >0.5 pt | X2 |
| X4 | Verify the exported SE hardsigmoid slope matches training in every runtime | — |
| X5 | Drop orientation and unwarping stages if the capture UI enforces framing | Product decision |

## Open questions the literature does not answer

- Does self-supervised pre-training help a CTC convnet under 10M parameters?
- Does feature distillation from a foundation teacher carry over to sub-5M
  convnets on text, as it does at ~10M on COCO?
- Does CR-CTC or the semantic guidance module help small conv recognizers?
- What does int8 cost a DB detector or a large-vocabulary CTC head? No primary
  study was found.
- Phone latency for PP-OCRv3 and later: none published.

Each of R5, R7, T5 and D1 would be a new data point, not a reproduction.
