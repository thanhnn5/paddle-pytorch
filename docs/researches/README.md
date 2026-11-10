# Edge OCR research: state and direction (survey, 2026-11-10)

Scope: lightweight OCR that runs on phones and other edge devices, with the
PaddleOCR (Baidu) line as the reference system. This is a literature survey only.
Nothing here was measured on our models; every expected gain is extrapolated
from the cited papers.

How it was produced: six Opus sub-agents each surveyed one track and wrote the
sourced notes in `notes/`. A seventh re-checked 15 load-bearing claims against
primary sources (`notes/90-verification-log.md`). This file is the supervisor's
synthesis; the experiment backlog is in [roadmap.md](roadmap.md).

| File | Content |
|---|---|
| [notes/01-paddleocr-lineage.md](notes/01-paddleocr-lineage.md) | PP-OCRv1 → v6, PaddleOCR 3.x, PaddleOCR-VL |
| [notes/02-text-detection.md](notes/02-text-detection.md) | Detection architectures, losses, edge backbones |
| [notes/03-text-recognition.md](notes/03-text-recognition.md) | Recognition architectures, CTC comeback, large vocabularies |
| [notes/04-loss-functions.md](notes/04-loss-functions.md) | CTC variants, guide heads, distillation objectives |
| [notes/05-pretraining-semisup.md](notes/05-pretraining-semisup.md) | Self-/semi-supervised learning, synthetic data, large-model teachers |
| [notes/06-edge-deployment-and-vlm.md](notes/06-edge-deployment-and-vlm.md) | Op/runtime compatibility, quantization, VLM-OCR catalogue |
| [notes/90-verification-log.md](notes/90-verification-log.md) | Fact-check of 15 claims, with evidence and URLs |
| [roadmap.md](roadmap.md) | Ordered experiment backlog for when GPU capacity is available |

## 1. The short version

1. **The edge recipe has been stable for six years.** A DB-style segmentation
   detector plus a CTC recognizer, both on convolutional backbones, is still
   what ships. No lightweight end-to-end spotter or DETR-style text detector
   has displaced it.
2. **Gains moved from architecture to training signal.** PP-OCRv1–v2 were
   compression tricks, v3–v4 refreshed the architecture, v5's jump came mostly
   from data and large-model teachers, and v6 went back to an efficiency-first
   backbone with three size tiers.
3. **Almost every useful recent idea is free at inference.** Guide heads,
   auxiliary heads, distillation and re-parameterization are all training-only.
   The deployed graph stays a plain feed-forward convnet.
4. **CTC recognizers are competitive again.** SVTRv2 (ICCV 2025) beats
   encoder-decoder models on accuracy and speed with a CTC head, and is far
   better on long text.
5. **Large models became teachers and labelers, not on-device models.** Compact
   document VLMs (~0.9–1.2B parameters) lead page-parsing benchmarks, but the
   only phone evidence found is ~22 s per page on CPU.
6. **For small models, labels and initialization matter more than the loss.**
   Real in-domain data beats much larger synthetic sets, and pre-training helps
   small models most when labels are scarce.

## 2. How the architecture changed

### 2.1 The PaddleOCR line

| Version | Detector | Recognizer | What drove the gain |
|---|---|---|---|
| v1 (2020) | DB, MobileNetV3, SE removed | CRNN: MobileNetV3 + BiLSTM + CTC | Pruning, quantization, augmentation |
| v2 (2021) | + CML distillation | PP-LCNet, + U-DML, Enhanced CTC | Distillation |
| v3 (2022) | LK-PAN teacher, RSE-FPN student | SVTR-LCNet, CTC guided by a SAR head (GTC), 48 px height | Architecture + guide head + pseudo-labels (UIM) |
| v4 (2023) | PP-LCNetV3, PFHead, dynamic shrink ratio | PP-LCNetV3 + SVTR neck, guide head switched to NRTR, data filtering (DF) | Architecture + data cleaning |
| v5 (2025) | PP-HGNetV2 server model distilled from the GOT-OCR2.0 encoder | Same family, one model for 5 text types, 18,383-entry dictionary | Data scale, VLM labeling and hard-case mining |
| v6 (2026) | LCNetV4 backbone, RepLKFPN neck, training-only auxiliary DB heads, Dice + Focal loss | LCNetV4 + LightSVTR neck, CTC with NRTR guide; tiny distilled from medium | New backbone, three tiers (1.5M / 7.7M / 34.5M total) |

Notes on the table:
- v6 is the first version to change the detection loss. Its detector uses no
  distillation; only the tiny recognizer is distilled.
- v5's accuracy came with a slower mobile pipeline; the docs attribute this to
  the larger dictionary.
- Accuracy numbers across versions come from different in-house test sets, so
  only same-table comparisons are valid. The v6 paper reports tiny detection at
  80.6 average Hmean against 75.2 for v5 mobile on its own set.
- Two things survived every generation: DB detection, and CTC-only inference
  with a training-only attention guide.

### 2.2 Detection

- **Segmentation still wins on edge.** The line runs PSENet → PAN → DB → DB++ →
  FAST. DB's contribution was making binarization trainable; later work mostly
  changed the neck, the label representation and post-processing.
- **Necks got large-kernel and re-parameterized.** LK-PAN (9×9) in v3, then
  RepLKFPN (7×7 dilated re-param depthwise) in v6.
- **Post-processing is moving into the graph.** FAST replaces PAN's pixel
  aggregation with max-pool dilation on GPU (−0.2 F, +13.9 FPS on Total-Text).
- **Transformer text detectors are not edge candidates.** DPText-DETR, DeepSolo
  and SRFormer are ResNet-50-class and report no mobile latency. No text
  detector built on RT-DETR, D-FINE or DEIM was found.
- **Foundation backbones entered as teachers.** DEIMv2, RT-DETRv4 and
  EdgeCrafter distil DINOv3 features into small detectors. The smallest DINOv3
  ConvNeXt is ~29M parameters, so it is a teacher, not a deployable backbone.

### 2.3 Recognition

- **Paradigm sequence:** CRNN+CTC → attention decoders (ASTER, SAR) →
  transformers (NRTR, SATRN, ViTSTR) → language-model-augmented (SRN, ABINet) →
  single visual model (SVTR) → parallel/permutation decoders (PARSeq, CPPD) →
  pretrain-driven large models (MAERec, CLIP4STR) → CTC again (SVTRv2).
- **SVTRv2** (19.8M parameters, CTC) scores 86.14 on Union14M against 85.17 for
  MAERec, at 143 vs 17 FPS. It adds three things:
  - multi-size resizing by aspect ratio (MSR), instead of squashing to one width;
  - a feature rearrangement module (FRM), kept at inference;
  - a semantic guidance module (SGM), used only in training, worth +2.28.
- **Long text separates the paradigms.** On the long-text benchmark PARSeq,
  CPPD and ABINet score about 0; SVTRv2-B scores 50.2.
- **Vocabulary size now dominates small models.** With 18,383 classes and a
  120-dim neck, the CTC classifier is ~2.2M parameters. That is likely half or
  more of a mobile recognizer (an estimate, not measured on our export).
- **Two-stage detection + recognition remains the edge default.**

### 2.4 Backbones and runtimes

- The direction is BatchNorm + re-parameterized depthwise convolutions with
  simple activations (PP-LCNet v3/v4, RepViT, MobileOne, MobileNetV4).
- ConvNeXt-V2 has the least NPU-friendly op mix in the survey: LayerNorm, GELU,
  7×7 depthwise and GRN. Whether this costs latency on the Apple Neural Engine
  is unmeasured; Apple's own guidance does not call LayerNorm or GELU
  unsupported.
- Qualcomm's NPU path needs static shapes and quantized (or fp16) models.
- PP-OCRv6 small was slower than v5 mobile on an M4 with ONNX Runtime (1.29 s
  vs 1.10 s). No phone latency is published for v3 or later.

## 3. How the loss functions changed

### 3.1 Detection

- **DB paper:** BCE with 1:3 hard-negative mining on the probability and
  binary maps, L1 on the threshold map, weights α=1, β=10.
- **PaddleOCR (and our `det_db_loss.py`):** Dice-based loss with hard-negative
  balancing on the shrink map ×5, Dice on the binary map, L1 on the threshold
  map ×10. No published ablation compares the two weightings.
- **PP-OCRv4:** dynamic shrink ratio in label generation (0.4 → 0.6 over
  training), +1.27 Hmean in its ablation.
- **PP-OCRv6:** Dice + Focal (α=0.25, γ=2.5) on the shrink map, plus the same
  loss on training-only auxiliary heads at P2/P3/P4 (weights 0.4/0.3/0.2).
- **Distillation for detectors:**
  - CML (v2–v4): the student learns from a teacher's dilated shrink map and
    from a peer student.
  - RT-DETRv4: align one deep feature through a projector; cosine similarity
    beat MSE (53.7 vs 52.7 AP), and aligning backbone stages gave no gain.
  - EdgeCrafter: adapting the teacher to the task first is worth +0.8 AP, and a
    larger teacher was worse (DINOv3-L 52.6 vs B 54.3 student AP).

### 3.2 Recognition

- **Guide heads.** CTC is trained alongside an attention head that is dropped
  at inference: SAR in v3, NRTR from v4. SVTRv2's SGM beat this approach in a
  controlled ablation (86.14 vs 84.33 on Union14M, 80.03 vs 75.70 on the
  occluded set).
- **CTC regularizers.**
  - DCTC (AAAI 2024) adds a per-frame self-distillation term from the model's
    own CTC alignment: +2.6 / +2.1 on CRNN (English / Chinese), +0.5 to +1.1 on
    SVTR, no teacher needed.
  - EnCTC and Focal-CTC did not reproduce in DCTC's re-implementation (EnCTC
    −4.5 on Chinese CRNN).
  - CR-CTC (ICLR 2025) regularizes two augmented views to agree. It is
    speech-only so far; no OCR application was found.
- **Distillation.** DML / U-DML in v2–v3, then DKD on the guide head plus
  blank-masked, time-averaged CTC-logit distillation in v4. Naive frame-level
  distillation between CTC models with different alignments can hurt (68.7 →
  64.1 on Chinese CRNN in DCTC's test).

## 4. How pre-training and semi-supervised learning help

- **Real data first.** 276K real crops roughly match 16M synthetic crops
  (Baek et al., CVPR 2021).
- **Self-supervised pre-training helps small models most, when labels are
  scarce.**
  - DiG on real labeled data: ViT-Tiny +11.5 vs ViT-Base +5.8.
  - With 1% of labels (ViT-Small): scratch 5.2, DiG 62.9, SSM 66.0.
  - MAE-style pre-training goes the other way (MAERec: larger models gain more).
  - No paper tests these methods on a CTC convnet under 10M parameters. The
    only evidence at that size is PP-OCRv3: rotation pretext +0.6.
- **Semi-supervised learning works for small CTC models, but the method
  matters.** On CRNN, pseudo-labeling gave +2.8 and pseudo-labeling with
  rotation pre-training +3.4, while Mean Teacher consistency gave −0.2.
- **Pseudo-labels need filtering.** PP-OCRv4's two-model confidence band
  (drop >0.95 from a weak model, <0.15 from a strong one) added +1.2 and cut
  training time. Multi-teacher agreement is the other published filter.
- **Large models as offline teachers.** PP-OCRv5 distilled the GOT-OCR2.0
  encoder into its detector and used ERNIE-4.5-VL for hard-case mining and for
  annotating handwriting. No published recipe distils a VLM directly into a
  small CTC recognizer; the pseudo-label route is the supported one.
- **Distillation as pre-training.** Feature distillation from a foundation
  model on unlabeled images gave about +5 AP at ~10M parameters in EdgeCrafter.
  That evidence is on COCO with ViT students, not text or sub-5M convnets.

## 5. Where the field is heading

- **Two product lines.** Baidu now maintains a classic pipeline for edge and
  CPU (PP-OCRv6) and a 0.9B VLM for page parsing on GPU (PaddleOCR-VL 1.0 → 1.5
  → 1.6). The VLM line still uses a classic layout detector and was
  bootstrapped with classic-pipeline labels.
- **Classic pipelines keep line-level OCR on device** because of latency,
  energy, binary size, bounding boxes and faithfulness. VLMs win on layout,
  tables, formulas and degraded text.
- **Expected next steps for edge models,** as a reading of the trend:
  - more VLM-generated and VLM-filtered training data;
  - training-only heads and distillation instead of bigger deployed models;
  - script-specific or tiered models to contain the vocabulary cost;
  - server-side VLM fallback for low-confidence cases.

## 6. What this means for this repo

Checked against the code during this survey:

- The ported PP-OCRv6 detection configs match the paper: Dice + Focal with
  γ=2.5 and auxiliary weights 0.4/0.3/0.2. The medium config uses `RepLKPAN`
  where the paper says RepLKFPN; this matches upstream's config.
- `det_db_loss.py` uses PaddleOCR's weighting (α=5, β=10), which is what
  Paddle-converted weights assume.
- `aux_distill.py` already uses cosine similarity, but aligns backbone stages 2
  and 3. RT-DETRv4 found backbone-stage alignment gave no gain over a single
  deep feature. This is the cheapest distillation ablation available.
- The recognizer config already has CTC + NRTR guide heads, and the codebase
  already has PP-OCRv4's CTC-logit distillation.

Open risks, none verified here:

- **Licensing.** ConvNeXt-V2 ImageNet weights are CC-BY-NC, including the
  timm-hosted copies. If the student was initialized from them, that needs a
  decision before production. DINOv3 weights use Meta's custom licence, which
  has no non-commercial clause but carries use restrictions and should get a
  legal review.
- **GRN in fp16.** ConvNeXt-V2's GRN sums squares over the whole feature map,
  which can exceed the fp16 range on large detector maps. This is an analysis,
  not an observed failure.
- **ConvNeXt-V2 on the Neural Engine.** Its op mix may fall back to CPU/GPU.

The ordered backlog is in [roadmap.md](roadmap.md).

## 7. Corrections applied after verification

The raw notes are kept as the agents wrote them. Where they disagree with the
verification log, the log is right:

| Topic | Raw note says | Verified |
|---|---|---|
| PP-OCRv6 paper (note 01) | No arXiv paper | arXiv 2606.13108 is the official paper; arXiv 2603.24373 is the PP-OCRv5 paper (CVPR 2026) |
| DCTC gain on SVTR (note 04) | +0.6 to +1.0 | +0.5 to +1.1 |
| Consistency regularization on CRNN (note 05) | +3.1 | Mean Teacher −0.2; pseudo-label +2.8, with rotation pre-training +3.4 |
| EdgeCrafter teacher adaptation (notes 02, 05) | +1.0 AP | +0.8 AP |
| RT-DETRv4 cosine alignment (note 02) | 53.5 AP | 53.7 AP |
| PaddleOCR DB shrink-map loss (note 02) | BCE ×5 | Dice-based with hard-negative balancing ×5 |
| FAST post-processing baseline (note 02) | Replaces Vatti unclip | Replaces PAN's pixel aggregation; FAST is arXiv-only |
| PP-OCRv5 dictionary size | Stated in the tech report | Not in the report; counted from `ppocrv5_dict.txt` (18,383 lines) |

Claims tagged `[UNVERIFIED]` inside the notes were not re-checked.
