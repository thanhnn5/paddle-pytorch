# 05 — Pre-training, self-/semi-supervised learning, synthetic data and large-model teachers for small OCR models

Survey date: 2026-11-10. Scope: how these techniques help OCR, and whether the gains carry over to **small** models (detector under 5M parameters, CTC recognizer under 10M).

Conventions:
- Every number links to the source it came from. Gains marked "(computed)" are differences I took from that source's tables.
- `[UNVERIFIED]` marks claims I could not confirm from a primary source in this session (memory, secondary summaries, or abstract-level wording only).
- "Avg" means the averaged word accuracy over the paper's benchmark set. Benchmark sets differ between papers, so compare numbers within a paper, not across papers.

---

## 1. Self-supervised pre-training for text recognition (STR)

### 1.1 Method notes

**SeqCLR (CVPR 2021, arXiv 2012.10873).** Sequence-to-sequence contrastive learning: each feature map is split into "instances" (frames), and the contrastive loss is computed over them. The abstract reports large gains over supervised training when labels are reduced, and SOTA on handwriting when fine-tuned on 100% of labels. The abstract gives no numbers. [source](https://arxiv.org/abs/2012.10873)

**PerSec (AAAI 2022).** Hierarchical contrastive learning on low-level stroke features and high-level semantic features. Pre-trained on **UTI-100M**, 100M unlabeled text images covering 5 scenes and 4 languages. Tested with both CTC and attention decoders. [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/20062). Per-model numbers were not extracted `[UNVERIFIED]`.

**SimAN (CVPR 2022, arXiv 2203.10492).** Generative pretext task: reconstruct an augmented patch using a neighbouring patch from the same text line as the style reference (similarity-aware normalization). [arXiv](https://arxiv.org/abs/2203.10492v2). Numbers not extracted `[UNVERIFIED]`.

**DiG (ACM MM 2022, arXiv 2207.00193).** Contrastive learning plus masked image modeling (MIM) on ViT. This is the most useful paper here on model size, because it reports gains for Tiny, Small and Base. [ar5iv](https://ar5iv.labs.arxiv.org/html/2207.00193)

Pre-training data:
- Unlabeled real: CC-OCR, about **15.77M** crops.
- Synthetic labeled (STD): 17M.
- Real labeled (ARD): 2.78M.

Average accuracy, scratch vs DiG pre-training ([Table 5](https://ar5iv.labs.arxiv.org/html/2207.00193); gains computed):

| Size (params as reported) | Fine-tune data | Scratch | DiG | Gain |
|---|---|---|---|---|
| ViT-Tiny (20M) | STD | 68.7 | 75.3 | **+6.6** |
| ViT-Small (36M) | STD | 73.6 | 78.8 | +5.2 |
| ViT-Base (52M) | STD | 75.3 | 79.7 | +4.4 |
| ViT-Tiny (20M) | ARD | 65.5 | 77.0 | **+11.5** |
| ViT-Small (36M) | ARD | 73.4 | 80.7 | +7.3 |
| ViT-Base (52M) | ARD | 76.4 | 82.2 | +5.8 |

With few labels (ViT-S, a fraction of ARD), [Table 3](https://ar5iv.labs.arxiv.org/html/2207.00193) gives:

| ARD fraction | Scratch | DiG |
|---|---|---|
| 1% (27.8K) | 5.2 | 62.9 |
| 10% (278K) | 52.3 | 73.5 |
| 100% | 73.4 | 80.7 |

The "20M" parameter count for ViT-Tiny is the whole recognizer as the paper reports it, including the decoder. It is not a bare ViT-Ti encoder (about 5.5M).

**MaskOCR (arXiv 2206.00311).** MIM pre-trains the encoder on unlabeled real text. The decoder is then pre-trained with "masked image-language modeling" on rendered text while the encoder stays frozen. The abstract evaluates on Chinese and English. [arXiv](https://arxiv.org/abs/2206.00311). Numbers not extracted `[UNVERIFIED]`.

**CCD (ICCV 2023, arXiv 2211.00288).** Character-level self-distillation:
- A self-supervised character segmentation module finds character regions in unlabeled real images.
- Character-level pairs are kept aligned across two augmented views using the known transformation matrix. This allows strong geometric augmentation.
- The abstract reports +1.38% average on recognition over prior SOTA. [arXiv](https://arxiv.org/abs/2211.00288), [code](https://github.com/TongkunGuan/CCD) (only ViT-S and ViT-B checkpoints are released).

**MAERec / Union14M (ICCV 2023, arXiv 2307.08723, "Revisiting STR: A Data Perspective").** [arXiv HTML](https://arxiv.org/html/2307.08723)
- Union14M has **4M labeled + 10.6M unlabeled** (Union14M-U) real crops.
- MAE pre-training on Union14M-U: 20 epochs, 75% mask ratio.
- Union14M-Benchmark: ViT-S **73.5 → 78.6 (+5.1)**, ViT-B **74.2 → 85.2 (+11.0)**.
- Common benchmarks: ViT-S +1.0, ViT-B +1.7.
- Here the larger model gained **more** from pre-training (Table 6).

**Symmetric Superimposition Modeling, SSM (IJCAI 2024, arXiv 2405.05841).** [arXiv HTML](https://arxiv.org/html/2405.05841v2)
- Each image is superimposed with its flipped copy, and the model reconstructs both the original and the flipped image.
- Pre-trained on Union14M-U (about 10M) and on OCR-CC (15.77M).
- Only ViT-Tiny and ViT-Small are tested.
- Union14M avg, ViT-S: scratch 79.8 → SSM 84.3. The text says +3.5, which conflicts with the table.
- Common benchmarks (OCR-CC pre-training, fine-tuned on ARD):

| Size | DiG | CCD | SSM |
|---|---|---|---|
| ViT-Tiny | 93.4 | 94.2 | 95.1 |
| ViT-Small | 94.7 | 95.6 | 96.4 |

- Low-label regime, ViT-S on 1% ARD: scratch 5.2 / DiG 62.9 / CCD 63.7 / **SSM 66.0**. On 10%: 52.3 / 73.5 / 78.2 / **80.8**.

**LMIM (CVPR 2025 per index listings, arXiv 2503.18746).** Linguistics-aware MIM: a separate branch feeds linguistic information into MIM decoding. Pre-trained on Union14M-U. The authors also release an **11M unlabeled Chinese text** set. [arXiv](https://arxiv.org/pdf/2503.18746). Numbers not extracted `[UNVERIFIED]`.

**TextRotNet in PP-OCRv3 (arXiv 2206.03001).** This is the only self-supervised result in the survey on a **production-size CTC model** (SVTR-LCNet, 12M):
- Self-supervised rotation pre-training on "a large amount of unlabeled text line data". Quantity and source are not given.
- Gain: **+0.6** accuracy (76.3 → 76.9) on top of GTC and TextConAug.
- [ar5iv Table 4](https://ar5iv.labs.arxiv.org/html/2206.03001)

**"What if we only use real datasets" (CVPR 2021, arXiv 2103.04400).** On CRNN (a small CTC model) vs TRBA, gains over Baseline-real + Aug ([ar5iv](https://ar5iv.labs.arxiv.org/html/2103.04400)):

| Method | CRNN | TRBA |
|---|---|---|
| RotNet | +1.3 | +0.09 |
| MoCo | +0.8 | −0.8 |
| PL + RotNet | **+3.4** | +1.8 |

Self-supervised pre-training helped the small CTC model **more** than the larger attention model.

### 1.2 Do small models benefit?

- **Evidence that small models gain more.**
  - DiG's gain is largest at ViT-Tiny: +6.6 on synthetic and +11.5 on real-labeled data, vs +4.4 / +5.8 at Base.
  - In Baek et al., RotNet and pseudo-labeling help CRNN more than TRBA.
  - Reading: smaller models are more data-starved, so a good initialization matters more for them.
- **Evidence that large models gain more.** MAERec (MAE on Union14M-U) gains +11.0 at ViT-B vs +5.1 at ViT-S on the hard Union14M benchmark.
  - Reading: MAE-style pure reconstruction needs capacity to pay off.
  - Contrastive or discriminative objectives (DiG, CCD) and SSM appear friendlier to small models.
- **Gap in the evidence: sub-10M CTC convnets.** Almost all of these papers use ViT encoders of 20–36M total parameters with attention decoders. The only data points on sub-10M CTC convnets are:
  - PP-OCRv3 TextRotNet: +0.6 on a 12M model.
  - Baek et al. on CRNN: about +1.3, parameter count not reported.
  - No paper found reports DiG/CCD/SSM on SVTR-T or a <10M CTC model `[gap]`.
- **Where pre-training pays most: small label budgets.** At 1–10% of labels, pre-training gives tens of points; at 100%, it gives a few points (DiG and SSM tables above). This regime matches the POD case: few labels, many unlabeled images.

---

## 2. Self-/weakly-supervised pre-training for detection and spotting

**STKM (CVPR 2021).**
- A CNN encoder plus a self-attention decoder is pre-trained on SynthText with only **image-level** text labels.
- The backbone then transfers to any detector.
- Reported **+5.89% F** for EAST on ICDAR15.
- [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wan_Self-Attention_Based_Text_Knowledge_Mining_for_Text_Detection_CVPR_2021_paper.html), [code](https://github.com/CVI-SZU/STKM)
- The STKM paper is CVPR 2021, not 2024.

**oCLIP (ECCV 2022, arXiv 2203.03911).**
- Weakly supervised vision-language pre-training from partial text transcriptions, with no boxes.
- Transferring to detection and spotting networks gives **+2.5% / +4.8% F** on ICDAR2019-LSVT weak annotations, and +3.2% / +1.3% on Total-Text / CTW1500 over other pre-training methods.
- [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1958_ECCV_2022_paper.php), [code](https://github.com/bytedance/oclip)
- The backbone is ResNet-50, far above the 5M budget.

**ODM (CVPR 2024).** Text-image further-alignment pre-training for scene text detection and spotting. [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Duan_ODM_A_Text-Image_Further_Alignment_Pre-training_Approach_for_Scene_Text_CVPR_2024_paper.pdf). Numbers not extracted `[UNVERIFIED]`.

**TCM (CVPR 2023, arXiv 2302.14338).** Plugs a CLIP ResNet-50 and a frozen CLIP text encoder into DBNet, PAN or FCENet. [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.14338)
- Few-shot: about **+22% average F** with only 10% of labels.
- Domain adaptation (DBNet → TCM-DBNet): ST→IC15 64.0 → 76.7; IC13→TD500 53.8 → 65.1. The paper states an average of +8.2, but the table values look larger; check before citing.
- Full data, DBNet: IC15 87.3 → 89.2.
- Use for us: as a **teacher or pseudo-labeller**, not as an edge model.

**Vision foundation backbones (DINOv2 / DINOv3 / SAM).**
- **DINOv3** releases these weights, all distilled from ViT-7B on LVD-1689M ([repo](https://github.com/facebookresearch/dinov3)):
  - ViT-S/16 (21M), ViT-S+ (29M), ViT-B (86M), ViT-L (300M), ViT-H+ (840M).
  - **ConvNeXt-T/S/B/L (29M / 50M / 89M / 198M)**.
  - The smallest ConvNeXt (29M) is still about 6× over a 5M detector budget. So DINOv3 is realistically a **teacher**, or a source for feature-distillation targets, not the deployed backbone.
  - The DINOv3 License is not Apache/MIT; check its terms for commercial use `[check]`.
- **SAM / SAM2 encoders and DINOv2 as text-detection backbones.** No primary text-detection result was confirmed this session `[UNVERIFIED]`.

**Semi-supervised spotting: SemiETS (CVPR 2025, arXiv 2504.09966).**
- Teacher-student pseudo-labelling with progressive sample assignment and mutual mining between the detection and recognition branches.
- Total-Text end-to-end H-mean: **+8.7 / +5.6 / +2.6** over prior SSL SOTA at **0.5% / 1% / 2%** labels.
- Also +2.0 over a fully supervised spotter.
- [arXiv](https://arxiv.org/pdf/2504.09966)

---

## 3. Semi-supervised STR / detection and real-vs-synthetic data

### 3.1 Real vs synthetic data

**Baek et al. 2021 ([ar5iv](https://ar5iv.labs.arxiv.org/html/2103.04400)).**
- 276K real labeled crops, only **1.7%** of MJ+ST's 16M, roughly match synthetic training:
  - CRNN: 74.8 (real) vs 75.8 (synthetic).
  - TRBA: 86.6 (real) vs 85.7 (synthetic).
- A synthetic subset of equal size (277K) is about **10 points worse** than real: CRNN 65.1, TRBA 75.9.

**Union14M ([arXiv HTML](https://arxiv.org/html/2307.08723)).**
- Synthetic-trained models average 87.03% on common benchmarks but **66.53%** on Union14M-L.
- Training on Union14M-L instead of synthetic data: **+3.9** on common benchmarks, **+19.6** on Union14M-Benchmark.
- MAERec-S on Union14M-Benchmark:

| Training | Accuracy |
|---|---|
| Synthetic fine-tuning | 46.0 |
| Union14M-L fine-tuning | 73.5 |
| Union14M-U pre-training + Union14M-L fine-tuning | 78.6 |

**Takeaway.** A modest amount of real in-domain data beats tens of millions of generic synthetic crops. In-domain unlabeled data stacks further gains on top.

### 3.2 Semi-supervised STR

**Pseudo-labeling and Mean Teacher (Baek et al.).**
- Pseudo-labeling: **+2.8 CRNN**, +1.7 TRBA.
- Mean Teacher slightly hurt both models (−0.2 / −0.4).
- [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.04400)

**Zheng et al., "Pushing the performance limit…" (CVPR 2022, arXiv 2204.07714).**
- Character-level consistency regularization in a BYOL-style asymmetric setup.
- Training data: synthetic labels + **~10.5M** real unlabeled crops (Places2, OpenImages, ImageNet).
- Gains: **CRNN 82.8 → 85.9 (+3.1)**, MORAN +1.7, HGA +2.5, TRBA 91.5 → 94.5 (+3.0).
- [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.07714)
- Small CTC models gain as much as large attention models.

**Seq-UPS (WACV 2023, arXiv 2209.00641).**
- Pseudo-labels come from beam search; uncertainty is estimated with an MC-dropout ensemble at character and word level.
- The motivation is that "high-confidence" pseudo-labels are often wrong, so a plain confidence threshold is unreliable.
- [arXiv](https://arxiv.org/abs/2209.00641v2). Numbers not extracted `[UNVERIFIED]`.

**SoftCTC (arXiv 2212.02135).**
- A CTC loss over multiple transcription variants, which removes the need for confidence-filtering of pseudo-labels.
- Matches a carefully tuned filtering pipeline on handwriting.
- [arXiv](https://arxiv.org/abs/2212.02135)
- Directly applicable to a CTC student fed by **multiple teachers' transcripts**.

**ViSu (NeurIPS 2024, arXiv 2411.15585).**
- Training data: synthetic + Union14M-U unlabeled, no human labels.
- Adds online generation of background-free, varied-style glyph samples and a Character Unidirectional Alignment loss.
- 24.4M model: common benchmarks 92.5 → **94.7**, Union-B 62.6 → **70.9**.
- Re-run on the same data, Zheng's TRBA-cr reaches only 54.7 on Union-B.
- [ar5iv](https://ar5iv.labs.arxiv.org/html/2411.15585)

**Cloze Self-Distillation (arXiv 2503.16184).**
- A teacher trained on noisy real labels re-labels the training set with cloze-style (context) predictions; the student is trained on those pseudo-labels plus a KL term.
- Gains are small at full data: ViT-T 91.6 → 91.9, ViT-B 92.0 → 92.5.
- Gains are large with little data: **10% of real data: 86.9 → 89.1**.
- The paper also finds that scaling the **decoder** beats scaling the encoder per parameter.
- [arXiv HTML](https://arxiv.org/html/2503.16184)

### 3.3 PP-OCR data strategies

**PP-OCRv3 UIM (unlabeled images mining).**
- A high-accuracy recognizer pseudo-labels unlabeled images, and high-confidence samples are kept.
- Data: 810K LSVT unlabeled images.
- Gain: **+1.0** (78.4 → 79.4) on the 12M SVTR-LCNet.
- The teacher model and threshold are not given.
- U-DML (same-architecture mutual distillation) adds **+1.5**.
- [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.03001)
- UIM is **not** in the PP-OCRv2 paper; it was introduced in v3.

**PP-OCRv4 DF (data filter).**
- A model trained on a subset scores the full set. Samples with confidence **>0.95** (redundant) and **<0.15** (too hard or noisy) are dropped.
- The training set shrank from tens of millions to millions, and training time fell from 2 weeks to 5 days.
- Accuracy: **71.5 → 72.7 (+1.2)**.
- [PaddleOCR docs](https://www.paddleocr.ai/latest/version2.x/ppocr/blog/PP-OCRv4_introduction.html). These are official docs, not a paper; the original repo docs were not cross-checked.

**PP-OCRv2.**
- CML distillation for the detector: ResNet18 teacher, two MobileNetV3 students. Hmean 0.759 → 0.789; with CopyPaste, 0.795.
- U-DML for the recognizer: +4.6 (0.693 → 0.739).
- [ar5iv](https://ar5iv.labs.arxiv.org/html/2109.03144)

### 3.4 Semi-supervised detection for small detectors

- No primary source found this session for semi-supervised **lightweight** text detection, such as a DBNet-MobileNet trained with pseudo-boxes. SemiETS (spotting) and the TCM few-shot results are the closest `[gap]`.

---

## 4. Large-model teachers and auto-labeling

**PP-OCRv5 / PaddleOCR 3.0 tech report (arXiv 2507.05595).** [arXiv HTML](https://arxiv.org/html/2507.05595)
- Detection uses **feature distillation from the GOT-OCR2.0 visual encoder** into a PP-HGNetV2 student.
- Detection hard cases are mined by comparing against **ERNIE-4.5-VL-424B-A47B**.
- Recognition: traditional models are combined with ERNIE-4.5-VL to "automatically annotate and filter" handwriting samples, including synthesized rare characters.
- More recognition data comes from PDFs and e-books, filtered by edit distance.
- Reported result: handwriting error rate −26%.
- **Not reported:** data volumes, and separate gains for the mobile vs server models.

**PaddleOCR-VL (arXiv 2510.14528).**
- Pseudo-labels come from an expert pipeline (PP-StructureV3) and are then refined by large multimodal models. The student is a 0.9B VLM, not a CTC model.
- [arXiv](https://arxiv.org/html/2510.14528v1)

**Multi-model agreement.**
- **Consensus Entropy (arXiv 2504.11101; CVPR 2026 poster):** an agreement-entropy score across several VLM OCR outputs. It improves quality-verification F1 by **42.1%** over VLM-as-judge, needs no training, and works for filtering. [arXiv](https://arxiv.org/abs/2504.11101), [CVPR](https://cvpr.thecvf.com/virtual/2026/poster/37103)
- **TeleOCR (arXiv 2608.12898):** reportedly keeps the pseudo-label with the highest edit-distance consensus across models, above a threshold. Seen only via a search summary `[UNVERIFIED]`.

**What is missing: a direct recipe.** No paper found that distills a VLM's OCR ability (GOT-OCR, Qwen-VL, InternVL, Florence-2, dots.ocr) into a **small CTC model** and reports the gain `[gap]`. The closest evidence chain is:
1. Teacher pseudo-labels plus confidence or agreement filtering (UIM, DF, Consensus Entropy).
2. Soft or multiple-variant CTC targets (SoftCTC).
3. Cloze self-distillation for label noise.

**LLM-based text correction for label cleaning.** No primary OCR training result confirmed `[UNVERIFIED]`. PP-OCRv5's ERNIE-VL "annotate and filter" step is the nearest documented use.

---

## 5. Synthetic data generation

**Rendering-based generators.**
- **MJSynth + SynthText:** about 16M crops in total; the standard baseline (Baek et al.).
- **SynthTIGER (ICDAR 2021, arXiv 2107.09313):** a unified renderer with length and character long-tail balancing. Its abstract claims better STR than MJ+ST combined. [arXiv](https://arxiv.org/abs/2107.09313). Margins not extracted `[UNVERIFIED]`.
- **TextRecognitionDataGenerator / text_renderer:** PP-OCRv2 uses a modified text_renderer, with **16M synthetic + 1.9M real** for recognition. [ar5iv](https://ar5iv.labs.arxiv.org/html/2109.03144)

**Diffusion-based generators.**
- **TextSSR (ICCV 2025, arXiv 2412.01137)** releases TextSSR-F (3.55M screened instances).
  - v1 claims training on it lands within 3.7% of real-data training and beats 4M existing synthetic samples.
  - v2 claims further gains when mixed with real data.
  - [arXiv](https://arxiv.org/html/2412.01137v1), [ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Ye_TextSSR_Diffusion-based_Data_Synthesis_for_Scene_Text_Recognition_ICCV_2025_paper.html)
  - The v2 numbers were not verified.
- **ViSu's online generation:** background-free glyph samples with varied styles, created to cover character morphology that MJ/ST lacks (§3.2).
- **TextDiffuser-2, AnyText/AnyText2, GlyphControl, SceneVTG:** primarily text-rendering or editing models. No measured gain as **recognizer training data** was confirmed this session `[UNVERIFIED]`.

**Implication for POD.**
- Synthetic data is useful for:
  - charset coverage (rare characters, digits/barcode-like strings, label templates);
  - detection CopyPaste. PP-OCRv2 CopyPaste adds +0.6 Hmean.
- It is not a substitute for real in-domain data. On equal budgets, real data is about 10 points better (§3.1).
- Domain-targeted synthesis for parcel labels or receipts: no primary benchmark found `[gap]`.

---

## 6. Distillation as pre-training for small backbones

**EdgeCrafter (arXiv 2603.18739, Mar 2026).** [arXiv HTML](https://arxiv.org/html/2603.18739)
- Teacher: DINOv3-S/B, first adapted to COCO detection ("ECTeacher") and then frozen.
- Students: compact ViTs (embed dim 192–384) trained by **feature-only L2 distillation** on ImageNet-1K + COCO images. The distillation loss uses no labels; teacher adaptation does use COCO labels.
- ECDet-S (10M): **51.7 AP**.
- Ablation (ECDet-S): random init 46.6, Co-DETR teacher 50.4, generic DINOv3-S teacher 50.7, **task-adapted teacher 51.7**.
- Distillation data ablation: ImageNet only 54.1 vs **+COCO 54.3** (ECDet-M). Adding in-domain images helps slightly.
- The distilled backbone transfers to pose estimation: random 58.6 → **68.4 AP**.
- Interpretation: about **+5 AP from distillation alone**, and +1 more from adapting the teacher to the task first.

**DEIMv2 (arXiv 2509.20787).**
- DINOv3 features plus a parameter-free Spatial Tuning Adapter for the S–X variants.
- The ultra-light variants use pruned **HGNetv2-B0**, not DINOv3:
  - Pico: **1.5M params, 38.5 AP**.
  - Femto: 0.96M, 31.0 AP.
  - Atto: 0.49M, 23.8 AP.
  - The Femto/Atto parameter readings are inferred from the table layout.
- [arXiv](https://arxiv.org/pdf/2509.20787)

**RT-DETRv4 (arXiv 2510.25257).**
- A training-only Deep Semantic Injector aligns the detector's deepest feature map with a VFM teacher (e.g. DINOv3-ViT-B), with gradient-guided weighting of the transfer.
- Zero inference cost.
- COCO AP S/M/L/X ≈ 49.8 / 53.7 / 55.4 / 57.0; versions differ slightly.
- [arXiv](https://arxiv.org/abs/2510.25257)

**Generic foundation-model distillation.**
- **DINOv3:** distilled ConvNeXt variants (§2). The ConvNeXt-T student still has 29M params.
- **Proteus (ICLR 2025, arXiv 2407.10366):** distils DINOv2/CLIP into smaller ViTs on **ImageNet-1K only** (1.2M images) with token-, patch- and feature-level losses. Proteus-L matches DINOv2-L across 19 benchmarks. [arXiv](https://arxiv.org/abs/2407.10366)
  - Evidence that **~1M unlabeled images is enough** for feature distillation.
- **TinyViT (ECCV 2022, arXiv 2207.10666):** pre-training distillation with teacher logits **sparsified and stored offline**. That is useful when there is no big GPU: run the teacher once, train the student cheaply. 21M → 84.8% IN-1k. [arXiv](https://arxiv.org/abs/2207.10666)
- **AM-RADIO / RADIOv2.5 (arXiv 2412.07679):** multi-teacher distillation (CLIP, DINO, SAM). Its fixes include multi-resolution training, mosaic augmentation and teacher-loss balancing. Released variants are B/L/H/g; there is no small-model data. [arXiv](https://arxiv.org/abs/2412.07679)

**Data-efficient adaptation.**
- **Frozen backbone + few-shot fine-tuning:** TCM (§2) shows strong few-shot detection from a CLIP backbone (+22 F at 10% labels).
- **Active learning for OCR:** no primary result confirmed this session `[gap]`.

---

## 7. Summary table

| Method | Year | Venue | Type | Unlabeled / extra data | Model size tested | Reported gain |
|---|---|---|---|---|---|---|
| SeqCLR | 2021 | CVPR | SSL contrastive (STR) | handwriting + scene text | CNN/LSTM | qualitative in abstract [UNVERIFIED nums] |
| Baek et al. (real-only) | 2021 | CVPR | data / PL / RotNet | 276K real lbl + unlabeled | CRNN, TRBA | PL+RotNet: CRNN +3.4, TRBA +1.8 |
| STKM | 2021 | CVPR | weakly-sup det pretrain | SynthText image-level labels | ResNet + EAST | +5.89 F (EAST, IC15) |
| PerSec | 2022 | AAAI | SSL contrastive (STR) | UTI-100M | CTC + attn | [UNVERIFIED nums] |
| SimAN | 2022 | CVPR | SSL generative (STR) | unlabeled real | — | [UNVERIFIED nums] |
| DiG | 2022 | ACM MM | SSL contrastive + MIM | 15.77M CC-OCR | ViT-T/S/B (20/36/52M) | Tiny +11.5, Base +5.8 (ARD) |
| Zheng et al. (CR) | 2022 | CVPR | semi-sup consistency | 10.5M real unlbl | CRNN…TRBA | CRNN +3.1, TRBA +3.0 |
| oCLIP | 2022 | ECCV | weakly-sup V-L pretrain (det/spot) | web / LSVT weak | ResNet-50 | +2.5 F det, +4.8 spot |
| PP-OCRv3 TextRotNet / UIM | 2022 | arXiv | SSL / pseudo-label | 810K LSVT unlbl (UIM) | 12M CTC | +0.6 / +1.0 |
| MaskOCR | 2022 | arXiv | MIM + rendered text | unlabeled real | — | [UNVERIFIED nums] |
| Seq-UPS | 2023 | WACV | semi-sup PL selection | — | — | [UNVERIFIED nums] |
| SoftCTC | 2022 | arXiv | semi-sup soft CTC | handwriting | CTC | ≈ tuned filtering |
| TCM | 2023 | CVPR | CLIP → detector | CLIP | CLIP-R50 + DBNet | +22 F avg at 10% labels |
| CCD | 2023 | ICCV | SSL char-level distill | URD / OCR-CC | ViT-S/B | +1.38 avg over SOTA |
| MAERec / Union14M | 2023 | ICCV | MAE + real data | 10.6M Union14M-U | ViT-S/B | S +5.1, B +11.0 (Union-B) |
| PP-OCRv4 DF | 2023 | docs | data filtering | — | ~10M CTC | +1.2 |
| SSM | 2024 | IJCAI | SSL superimposition | 10M / 15.77M | ViT-T/S | S: 1% lbl 5.2 → 66.0 |
| ViSu | 2024 | NeurIPS | semi-sup, no human labels | Union14M-U | 24.4M | Union-B +8.3 |
| TextSSR | 2025 | ICCV | diffusion synthetic | 3.55M synth | — | within 3.7% of real (v1) |
| LMIM | 2025 | CVPR (listed) | linguistics-aware MIM | Union14M-U + 11M Chinese | — | [UNVERIFIED nums] |
| Cloze Self-Distill | 2025 | arXiv | self-distill, label noise | — | ViT-T/S/B | +0.3–0.5; 10% data +2.2 |
| SemiETS | 2025 | CVPR | semi-sup spotting | 98–99.5% unlbl | spotter | +8.7 H at 0.5% labels |
| PP-OCRv5 | 2025 | arXiv | VLM labeling + VFM distill | ERNIE-VL, GOT-OCR enc. | mobile / server | handwriting err −26% |
| Proteus | 2025 | ICLR | VFM distill on IN-1K | 1.2M images | ViT-S…L | ≈ DINOv2 oracle |
| DEIMv2 | 2025 | arXiv | DINOv3 features | — | 0.5M–50M | Pico 1.5M → 38.5 AP |
| RT-DETRv4 | 2025 | arXiv | VFM feature distill | — | S–X | SOTA, 0 inference cost |
| EdgeCrafter | 2026 | arXiv | task-specialized VFM distill | IN-1K + COCO imgs | ECDet-S 10M | +5.1 AP vs random |
| Consensus Entropy | 2025/26 | CVPR'26 | multi-VLM agreement | — | — | +42.1% F1 verification |

---

## 8. Practical recipes for POD

Setup:
- (a) A large pool of **unlabeled** parcel-label / proof-of-delivery (POD) photos.
- (b) A small labeled set.
- (c) Offline server models and VLMs as teachers.
- (d) No big local GPU.
- Targets: a detector **under 5M** parameters and a CTC recognizer **under 10M**.

Common principles from the evidence:
1. **Real in-domain data beats generic synthetic data** (§3.1). Spend teacher compute on turning unlabeled POD images into labels before generating synthetic data.
2. **Filter pseudo-labels** with confidence bands (DF: 0.15/0.95) or multi-teacher agreement (Consensus Entropy, edit-distance consensus). Plain high-confidence selection is unreliable (Seq-UPS).
3. **Small models gain most from initialization when labels are scarce** (DiG Tiny +11.5; SSM/DiG 1–10% tables). Unlabeled data should be used twice: once for pre-training or distillation, once for pseudo-labels.
4. **Teachers run once, offline.** Store their outputs (TinyViT-style) so student training fits on a modest GPU.

### Pipeline A — teacher auto-labeling + filtered self-training (lowest risk, cheapest)

1. **Detection pseudo-boxes.**
   - Run 1–2 strong server detectors over the unlabeled pool, e.g. PP-OCRv5 server det, a TCM/DBNet-R50, or the OCR mode of a VLM.
   - Keep boxes on which the detectors agree (IoU ≥ τ), and send disagreements to human review.
   - Basis: the SemiETS/TCM label-efficiency evidence, and PP-OCRv5's VLM hard-case mining.
2. **Recognition pseudo-labels.**
   - Crop and transcribe each box with ≥2 independent teachers, e.g. PP-OCRv5 server rec plus one VLM (Qwen-VL / InternVL / GOT-OCR).
   - Accept a crop when normalized edit distance = 0 or the agreement score is high (Consensus Entropy idea). Route the rest to humans or drop them.
   - Basis: UIM +1.0, DF +1.2, Consensus Entropy, PP-OCRv5's annotate-and-filter. Expected per-step gains on 10M-class models are around **+1 point each**.
3. **Data filter.** Drop crops the student already finds trivially easy (>0.95) and those that look like noise (<0.15). This also cuts training cost (PP-OCRv4 DF).
4. **Train students.**
   - Detector: train the small DB-style detector on human labels plus filtered pseudo-boxes. Add CML/DML distillation from the server detector (PP-OCRv2: +3 Hmean).
   - Recognizer: train the <10M CTC model with a GTC/NRTR guidance branch, U-DML (+1.5), and logit distillation from a server recognizer. SoftCTC is an option where teachers disagree.
5. **One self-training round.** Re-label with the improved student plus the teachers, then retrain (Baek: PL +2.8 on CRNN).

### Pipeline B — domain feature-distillation pre-training, then fine-tuning (best accuracy for small backbones)

1. **Backbone distillation on unlabeled POD images.**
   - Distil a frozen foundation teacher (DINOv3 ViT-S/B or ConvNeXt-T, optionally first adapted to text detection on the labeled set + pseudo-boxes) into the <5M detector backbone.
   - Use feature-only L2 alignment (EdgeCrafter: ECDet-S +5.1 AP over random init, +1.0 from task-adapting the teacher; adding in-domain images +0.2).
   - Precompute and store teacher features if GPU memory is tight (TinyViT-style).
   - Proteus suggests about 1M images is enough.
   - Caveats:
     - The EdgeCrafter evidence is on COCO with 10M+ ViTs, not <5M convnets or text.
     - The DINOv3 license must be checked.
2. **Recognizer encoder pre-training on unlabeled crops.**
   - Use detector crops from the unlabeled pool, with SSM or DiG/CCD-style pre-training.
   - Caveat: no evidence on <10M CTC convnets. For a CNN backbone, PP-OCRv3's TextRotNet (+0.6) is the proven cheap option.
3. Fine-tune both models on human labels plus Pipeline-A pseudo-labels. Optionally add RT-DETRv4-style auxiliary feature alignment during fine-tuning (zero inference cost).

### Pipeline C — semi-supervised consistency for the recognizer + targeted synthesis (when teacher labels are weak on the domain)

1. Train the CTC recognizer on human labels plus synthetic data focused on the domain charset: tracking numbers, postcodes, label fonts. Use a SynthTIGER/text_renderer-style generator with length and charset balancing. Diffusion synthesis (TextSSR-style) is optional for style diversity.
2. Add character-level consistency regularization on unlabeled POD crops (Zheng et al.: CRNN +3.1; ViSu: +8.3 on hard text with no human labels).
3. Apply Cloze self-distillation to clean noisy labels; the gain is largest with little data (+2.2 at 10%).

**Suggested order.** Pipeline A first, because it is cheap and its gains are proven on 10M-class PP-OCR models. Then add Pipeline B's backbone distillation for the detector, which fits the existing `convnext-femto-distill` direction. Use C only if VLM teachers prove unreliable on POD crops.

---

## Sources

- DiG — https://ar5iv.labs.arxiv.org/html/2207.00193 ; https://arxiv.org/abs/2207.00193 ; https://github.com/ayumiymk/DiG
- CCD — https://arxiv.org/abs/2211.00288 ; https://openaccess.thecvf.com/content/ICCV2023/html/Guan_Self-Supervised_Character-to-Character_Distillation_for_Text_Recognition_ICCV_2023_paper.html ; https://github.com/TongkunGuan/CCD
- Union14M / MAERec — https://arxiv.org/html/2307.08723
- SSM — https://arxiv.org/html/2405.05841v2 ; https://ijcai.org/proceedings/2024/0085.pdf
- LMIM — https://arxiv.org/pdf/2503.18746
- MaskOCR — https://arxiv.org/abs/2206.00311
- SeqCLR — https://arxiv.org/abs/2012.10873
- PerSec — https://ojs.aaai.org/index.php/AAAI/article/view/20062
- SimAN — https://arxiv.org/abs/2203.10492v2
- PP-OCRv2 — https://ar5iv.labs.arxiv.org/html/2109.03144
- PP-OCRv3 — https://arxiv.org/abs/2206.03001 ; https://ar5iv.labs.arxiv.org/html/2206.03001
- PP-OCRv4 docs — https://www.paddleocr.ai/latest/version2.x/ppocr/blog/PP-OCRv4_introduction.html
- PaddleOCR 3.0 / PP-OCRv5 — https://arxiv.org/html/2507.05595
- PaddleOCR-VL — https://arxiv.org/html/2510.14528v1
- What if we only use real datasets — https://ar5iv.labs.arxiv.org/html/2103.04400
- Pushing the performance limit (Zheng et al.) — https://ar5iv.labs.arxiv.org/html/2204.07714
- Seq-UPS — https://arxiv.org/abs/2209.00641v2
- SoftCTC — https://arxiv.org/abs/2212.02135
- ViSu — https://ar5iv.labs.arxiv.org/html/2411.15585
- Cloze Self-Distillation — https://arxiv.org/html/2503.16184
- SemiETS — https://arxiv.org/pdf/2504.09966
- STKM — https://openaccess.thecvf.com/content/CVPR2021/html/Wan_Self-Attention_Based_Text_Knowledge_Mining_for_Text_Detection_CVPR_2021_paper.html
- oCLIP — https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1958_ECCV_2022_paper.php ; https://github.com/bytedance/oclip
- ODM — https://openaccess.thecvf.com/content/CVPR2024/papers/Duan_ODM_A_Text-Image_Further_Alignment_Pre-training_Approach_for_Scene_Text_CVPR_2024_paper.pdf
- TCM — https://ar5iv.labs.arxiv.org/html/2302.14338
- DINOv3 — https://github.com/facebookresearch/dinov3
- EdgeCrafter — https://arxiv.org/html/2603.18739
- DEIMv2 — https://arxiv.org/pdf/2509.20787
- RT-DETRv4 — https://arxiv.org/abs/2510.25257
- Proteus — https://arxiv.org/abs/2407.10366
- TinyViT — https://arxiv.org/abs/2207.10666
- RADIOv2.5 — https://arxiv.org/abs/2412.07679
- SynthTIGER — https://arxiv.org/abs/2107.09313
- TextSSR — https://arxiv.org/html/2412.01137v1 ; https://openaccess.thecvf.com/content/ICCV2025/html/Ye_TextSSR_Diffusion-based_Data_Synthesis_for_Scene_Text_Recognition_ICCV_2025_paper.html
- Consensus Entropy — https://arxiv.org/abs/2504.11101 ; https://cvpr.thecvf.com/virtual/2026/poster/37103
- TeleOCR (secondary only) — https://arxiv.org/pdf/2608.12898
