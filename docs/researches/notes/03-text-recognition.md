# 03 — Text Recognition (STR): architecture evolution and edge deployability, 2015 → Nov 2026

Status: research notes for the supervisor to verify. Written 2026-11-10.

Conventions:
- Every numeric claim has an inline link to its source. Where the number comes from a table in *another* paper (for example the SVTR or SVTRv2 comparison tables), that paper is the link, not the original.
- `[UNVERIFIED]` means I did not confirm it from a primary source in this session. It may be from memory or from a secondary page.
- "Com" = average word accuracy over the 6 common English benchmarks (IIIT5k, SVT, IC13, IC15, SVTP, CUTE80). "U14M" = Union14M-L benchmark average. "LTB" = Long Text Benchmark (from SMTR). "OST" = occluded scene text. "BCTR" = the Chinese text recognition benchmark (scene / web / document / handwriting).
- The numbers below come from different training protocols (synthetic MJ+ST, real data, U14M-Filter, in-house sets). **Do not compare across protocols.** Each table states its protocol.

---

## 0. TL;DR for an edge CTC recognizer

1. **The production edge default is still CTC with an auxiliary attention branch used only during training.** PP-OCRv3 (GTC with SAR guidance), PP-OCRv4 (GTC-NRTR), PP-OCRv5, PP-OCRv6 and SVTRv2 (SGM) all follow this pattern. At inference they run one feed-forward pass plus a greedy CTC decode. ([PP-OCRv3](https://arxiv.org/html/2206.03001), [PP-OCRv4 doc](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html), [PaddleOCR 3.0 report](https://arxiv.org/html/2507.05595v1), [PP-OCRv6](https://arxiv.org/html/2606.13108), [SVTRv2](https://arxiv.org/html/2411.15858v2))
2. **SVTRv2 (ICCV 2025) reports that a CTC model beats encoder-decoder models when they are trained on the same real data.** SVTRv2-B (19.8M) scores U14M 86.14 at 143 FPS. Under the same protocol, MAERec-S (35.7M) scores 85.17 at 17.1 FPS and PARSeq (23.8M) scores 84.26 at 52.6 FPS. ([SVTRv2 Table 3](https://arxiv.org/html/2411.15858v2))
3. **Long text is where CTC wins structurally.** On LTB, PARSeq, CPPD and ABINet score **0.0**, while SVTRv2-B scores 50.2 ([SVTRv2](https://arxiv.org/html/2411.15858v2)). Fixed-max-length decoders cannot emit past their limit.
4. **For large-vocabulary mobile models, the classifier head is a large share of the parameters.** PP-OCRv5's dictionary has 18,383 entries (local copy: `torchocr/utils/dict/ppocrv5_dict.txt`). With a 120-dim neck output, the CTC linear layer alone is ≈2.2M parameters, out of a model with a 16 MB checkpoint. That is my estimate; details are in §5.
5. **PP-OCRv6 (June 2026) is the current Paddle edge line:** LCNetV4 MetaFormer backbone, CTC at inference, NRTR only in training. Recognition parameters are 1.1M / 5.2M / 19M for the tiny, small and medium tiers. ([PP-OCRv6](https://arxiv.org/html/2606.13108))

---

## 1. Timeline table

Edge-suitability is my judgement: **A** = single pass, CTC or linear head, under 10M params, convolution-friendly ops. **B** = single pass but heavier, or dominated by attention. **C** = autoregressive or large pretrained model, not practical on a phone CPU.

| Model | Year | Venue | Encoder | Decoder | Params | Key accuracy (protocol) | Single pass? | Edge |
|---|---|---|---|---|---|---|---|---|
| CRNN | 2015 (arXiv) / 2017 | TPAMI `[UNVERIFIED venue]` | VGG-style CNN + BiLSTM | CTC | 8.3M ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | IC13 91.1 / IC15 69.4 / Chinese Scene 53.4 (MJ+ST, CombBest reproduction) ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | Yes | A |
| Rosetta | 2018 | KDD `[UNVERIFIED venue]` | ResNet | CTC | 44.3M ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | IC15 71.2 ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | Yes | B |
| ASTER | 2018/2019 | TPAMI `[UNVERIFIED venue]` | TPS rectifier + ResNet + BiLSTM | AR attention | 27.2M ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | IC15 76.1 / CUTE 79.5 ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | No (AR) | C |
| NRTR | 2019 | ICDAR ([OpenOCR](https://github.com/Topdu/OpenOCR)) | CNN + Transformer encoder | AR Transformer | 31.7M, 160 ms ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | IC13 94.7 ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | No | C (used as the *training-only* guide branch in PP-OCRv4/v5/v6) |
| SAR | 2019 | AAAI `[UNVERIFIED venue]` | ResNet + LSTM | AR 2D attention | 57.5M, 120 ms ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | IC15 69.2, Chinese 62.5 ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | No | C (used as the GTC guide in PP-OCRv3) |
| RobustScanner | 2020 | ECCV `[UNVERIFIED]` | ResNet | AR attention + position branch | `[UNVERIFIED]` | — | No | C |
| SATRN | 2020 | CVPRW `[UNVERIFIED]` | 2D self-attention encoder | AR Transformer | `[UNVERIFIED]` | — | No | C |
| SRN | 2020 | CVPR `[UNVERIFIED venue]` | ResNet + Transformer | Parallel + semantic reasoning | 54.7M, 25.4 ms ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | IC15 82.7, Chinese 60.1 ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | Yes (parallel) | B |
| ViTSTR | 2021 | ICDAR `[UNVERIFIED venue]` | ViT (DeiT) | Linear (no decoder) | 85.5M, 11.2 ms ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | IC15 78.5 ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | Yes | B |
| ABINet | 2021 | CVPR `[UNVERIFIED venue]` | ResNet + Transformer | Parallel + iterative bidirectional LM | 36.7M, 51.3 ms ([SVTR T4](https://arxiv.org/pdf/2205.00159)); 36.9M ([SVTRv2 T3](https://arxiv.org/html/2411.15858v2)) | Com 95.83, U14M 75.72, **LTB 0.0** (U14M-Filter) ([SVTRv2](https://arxiv.org/html/2411.15858v2)) | Iterative | B/C |
| VisionLAN | 2021 | ICCV `[UNVERIFIED venue]` | ResNet + Transformer, language-aware training via masking | Parallel; LM dropped at inference | 32.8M, 28.0 ms ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | IC15 83.7 ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | Yes | B |
| PGNet (spotter) | 2021 | AAAI ([arXiv](https://arxiv.org/abs/2104.05458)) | FCN | PG-CTC on 2D maps | — | 46.7 FPS on Total-Text ([arXiv](https://arxiv.org/abs/2104.05458)) | Yes | B |
| TrOCR | 2021 | arXiv ("work in progress") ([arXiv](https://arxiv.org/abs/2109.10282)) | BEiT/ViT pretrained | AR text Transformer (wordpiece) | `[UNVERIFIED]` (roughly 62M to 558M from memory) | SOTA claims on printed, handwritten and scene text ([arXiv](https://arxiv.org/abs/2109.10282)) | No | C |
| **SVTR** | 2022 | IJCAI ([arXiv](https://arxiv.org/abs/2205.00159)) | Patch-embed (H/4) + local/global mixing, height merging | Linear CTC | T 6.03M / S 10.3M / B 24.6M / L 40.8M; T 4.5 ms on 1080Ti ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | SVTR-T: IC13 96.3, IC15 84.1, Chinese Scene 67.9. SVTR-L: Chinese 72.1 (MJ+ST) ([SVTR T4](https://arxiv.org/pdf/2205.00159)) | Yes | A (T) on GPU; slow on CPU-MKLDNN, see §4 |
| PP-OCRv3 rec (SVTR-LCNet) | 2022 | arXiv tech report ([arXiv](https://arxiv.org/abs/2206.03001)) | PP-LCNet (3 stages) + 2 SVTR global blocks, h=48 | CTC (+SAR GTC in training) | 12M ([PP-OCRv4 doc](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)) | 79.4% in-house, 7.6 ms CPU ([PP-OCRv3](https://arxiv.org/html/2206.03001)) | Yes | A |
| **PARSeq** | 2022 | ECCV ([arXiv](https://arxiv.org/abs/2207.06966)) | ViT-S | Permuted AR LM; AR, NAR or iterative refinement | 23.8M ([SVTRv2 T3](https://arxiv.org/html/2411.15858v2)) | 91.9 (synthetic) / 96.0 (real) on benchmarks ([arXiv](https://arxiv.org/abs/2207.06966)). Com 96.40, U14M 84.26, **LTB 0.0** (U14M-F) ([SVTRv2](https://arxiv.org/html/2411.15858v2)) | AR by default; NAR mode is single pass | B/C |
| PP-OCRv4 mobile rec | 2023 | docs ([PP-OCRv4](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)) | PP-LCNetV3 + Lite SVTR neck | CTC (+GTC-NRTR in training) | ~11M ([PP-OCRv4](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)); 10.5 MB ([module doc](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)) | 78.74% in-house; CPU 17.48 / 3.61 ms ([module doc](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)) | Yes | A |
| CPPD | 2023 (arXiv) / TPAMI 2025 per [OpenOCR](https://github.com/Topdu/OpenOCR) | SVTR-B | Parallel, with counting and ordering modules | 27.0M ([SVTRv2 T3](https://arxiv.org/html/2411.15858v2)) | About 8× faster than AR counterparts ([arXiv](https://arxiv.org/abs/2307.12270)). Com 96.40, U14M 81.91, **LTB 0.0** ([SVTRv2](https://arxiv.org/html/2411.15858v2)) | Yes | B |
| LISTER | 2023 | ICCV ([CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.html)) | FocalNet-B | Neighbor decoder (length-insensitive) | 51.1M ([SVTRv2 T3](https://arxiv.org/html/2411.15858v2)) | Com 96.03, U14M 77.64, LTB 36.3 ([SVTRv2](https://arxiv.org/html/2411.15858v2)) | Iterative over length | C |
| MAERec / Union14M | 2023 | ICCV ([arXiv](https://arxiv.org/abs/2307.08723)) | ViT pretrained with MAE on 10M unlabeled images | AR Transformer | 35.7M (ViT-S) ([SVTRv2 T3](https://arxiv.org/html/2411.15858v2)) | Com 96.36, U14M 85.17, LTB 9.8, 17.1 FPS ([SVTRv2](https://arxiv.org/html/2411.15858v2)) | No | C |
| CLIP4STR | 2023 (arXiv) / T-IP ([arXiv](https://arxiv.org/abs/2305.14014)) | CLIP ViT visual + cross-modal branches | Predict-and-refine (PARSeq-style) | 158M, 14.1 FPS ([SVTRv2 T6](https://arxiv.org/html/2411.15858v2)) | "SOTA on 13 STR benchmarks" ([arXiv](https://arxiv.org/abs/2305.14014)); Com 97.32 (synthetic pretrain → real fine-tune) ([SVTRv2 T6](https://arxiv.org/html/2411.15858v2)) | No | C |
| DTrOCR | 2023 | WACV 2024 ([arXiv](https://arxiv.org/abs/2308.15996)) | None (decoder-only) | Decoder-only generative LM (GPT-2 per my memory, `[UNVERIFIED]`) | `[UNVERIFIED]` | "outperforms SOTA by a large margin" ([arXiv](https://arxiv.org/abs/2308.15996)) | No | C |
| IGTR | 2024 | TPAMI ([arXiv](https://arxiv.org/abs/2401.17851)) | SVTR-style | Instruction-guided Q&A over character attributes | `[UNVERIFIED]` | Claims large gains on English and Chinese ([arXiv](https://arxiv.org/abs/2401.17851)) | Configurable | B |
| SMTR | 2024 | AAAI 2025 ([OpenOCR](https://github.com/Topdu/OpenOCR)) | SVTR-style | Sub-string matching cross-attention, iterated (handles any length) | `[UNVERIFIED]` | Introduces LTB; about half of LTB correct where PARSeq fails ([arXiv](https://arxiv.org/abs/2407.12317)) | No (iterative) | B/C |
| CDistNet, BUSNet, OTE, DPTR, FocalSVTR-CTC | 2023–2025 | various (see the [OpenOCR list](https://github.com/Topdu/OpenOCR)) | — | — | `[UNVERIFIED]` | Reproduced in OpenOCR; not individually verified here | — | — |
| **SVTRv2** | 2024 (arXiv) / ICCV 2025 ([arXiv](https://arxiv.org/abs/2411.15858)) | Multi-stage SVTR-like, MSR input, FRM | **CTC** (SGM in training only) | T 5.1M / S 11.3M / B 19.8M ([SVTRv2](https://arxiv.org/html/2411.15858v2)) | B: Com 96.57, U14M 86.14, LTB 50.2, OST 80.0, BCTR 83.31; 143 FPS (1080Ti, bs=1) ([SVTRv2](https://arxiv.org/html/2411.15858v2)) | **Yes** | A (T/S) |
| OpenOCR RepSVTR (mobile) | 2024–25 | repo ([OpenOCR doc](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md)) | RepSVTR (reparameterized) | CTC | `[UNVERIFIED]` | E2E 67.22% vs PP-OCRv4 62.77% on the Paddle competition leaderboard ([OpenOCR doc](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md)) | Yes | A |
| PP-OCRv5 mobile / server rec | 2025 | PaddleOCR 3.0 report ([arXiv](https://arxiv.org/abs/2507.05595)); "PP-OCRv5" paper ([arXiv 2603.24373](https://arxiv.org/abs/2603.24373)) listed at CVPR 2026 ([CVPR](https://cvpr.thecvf.com/virtual/2026/poster/38525)) | mobile: PPLCNetV3 ×0.95 + SVTR neck (dim 120). server: PPHGNetV2_B4 (repo configs) | CTC (+NRTR in training) | mobile 16 MB, server 81 MB ([module doc](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)); "~5M-parameter" ([2603.24373](https://arxiv.org/abs/2603.24373)); "0.07B" ([3.0 report](https://arxiv.org/html/2507.05595v1)) | mobile 81.29 / server 86.38 in-house; mobile CPU 21.20 / 5.32 ms ([module doc](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)) | Yes | A (mobile) |
| UniRec-0.1B | 2025 | arXiv 2512.21095 ([arXiv](https://arxiv.org/abs/2512.21095)) | `[UNVERIFIED]` | `[UNVERIFIED]` (likely AR) | ~0.1B | Text, formula and mixed content; "2–9× speedup" vs unnamed baselines ([arXiv](https://arxiv.org/abs/2512.21095)) | `[UNVERIFIED]` | C |
| MDiff4STR | 2025 (arXiv) / AAAI 2026 ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37370)) | — | Mask diffusion, 3 denoising steps | `[UNVERIFIED]` | "surpasses SOTA ARMs" (per search snippet of the [arXiv](https://arxiv.org/html/2512.01422v1) abstract) | 3 passes | B |
| **PP-OCRv6** tiny / small / medium rec | Jun 2026 | arXiv 2606.13108 ([arXiv](https://arxiv.org/html/2606.13108)) | LCNetV4 (MetaFormer: reparam DW-conv token mixer). Tiny has no neck; S/M use LightSVTR neck | CTC (+NRTR in training; tiny also distilled from medium) | 1.1M / 5.2M / 19M rec ([arXiv](https://arxiv.org/html/2606.13108)) | In-house W-Avg 73.5 / 81.3 / 83.2, vs PP-OCRv5 mobile 73.7 / server 78.1 on the *same* new set ([PP-OCRv6 doc](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)) | Yes | A |
| ScriptMoE | Sep 2026 | arXiv 2609.24058 ([arXiv](https://arxiv.org/abs/2609.24058)) | Shared visual encoder | Sparse MoE decoder (top-2 script experts + shared expert); CTC vs attention not stated | not stated | TextMuSS-Bench 82.06%. Swapping it in as PP-OCRv5's recognizer raises CC-OCR multilingual F1 from 65.71 to 80.89 ([arXiv](https://arxiv.org/abs/2609.24058)) | `[UNVERIFIED]` | `[UNVERIFIED]` |

---

## 2. Paradigm timeline (item 1)

1. **CRNN + CTC (2015 onward).** CNN, then BiLSTM, then a CTC head. It is fast, has no length limit, and runs in one pass. Its weaknesses are curved text and the lack of an explicit language model. In SVTR's reproduction table it has 8.3M params, 6.3 ms, and 69.4 on IC15 ([SVTR T4](https://arxiv.org/pdf/2205.00159)).
2. **Attention encoder-decoder (2018–2020): ASTER, SAR, RobustScanner.** Rectification (TPS) or 2D attention handles irregular text. Decoding is autoregressive, so latency grows with the number of characters (SAR 120 ms in [SVTR T4](https://arxiv.org/pdf/2205.00159)). These models also suffer attention drift and hallucination on long or unusual strings.
3. **Transformers (2019–2021): NRTR, SATRN, ViTSTR.** NRTR and SATRN are still AR. ViTSTR drops the decoder and uses a linear head, but it is 85.5M params ([SVTR T4](https://arxiv.org/pdf/2205.00159)).
4. **Language-model-augmented (2020–2021): SRN, ABINet, VisionLAN.** These add explicit (SRN, ABINet) or training-time implicit (VisionLAN) linguistic reasoning. VisionLAN is the precursor of the "train with language, infer vision-only" idea that GTC and SGM later reuse.
5. **Single visual model (2022): SVTR.** It removes the sequence model: a patch embedding at H/4, local and global mixing blocks, height-halving merges, then a linear CTC-style prediction ([SVTR](https://arxiv.org/pdf/2205.00159)). This is the architectural root of every PaddleOCR mobile recognizer since PP-OCRv3.
6. **Permutation and parallel decoders (2022–2023): PARSeq, CPPD, LISTER.** PARSeq trains an ensemble of AR LMs over permutations and can decode NAR or AR ([arXiv](https://arxiv.org/abs/2207.06966)). CPPD reaches AR-like accuracy with a parallel decoder at about 8× the speed ([arXiv](https://arxiv.org/abs/2307.12270)). LISTER targets length generalization ([CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.html)).
7. **Large pretraining (2021–2023): TrOCR, MAERec, CLIP4STR, DTrOCR.** Union14M showed that 13 models average only 66.53% on the 4M labeled Union14M images, against near-saturation on the common benchmarks ([arXiv](https://arxiv.org/abs/2307.08723)). Pretraining (MAE, CLIP, LMs) buys robustness at 35–158M+ params ([SVTRv2 T3/T6](https://arxiv.org/html/2411.15858v2)).
8. **Recent (2024–2026).**
   - The Fudan / OpenOCR line (Du, Chen et al.) includes SVTRv2 (CTC, ICCV 2025), SMTR (long text, AAAI 2025), IGTR (TPAMI), CPPD (TPAMI 2025), MDiff4STR (mask diffusion, AAAI 2026), UniRec-0.1B (text plus formula), OpenDoc-0.1B, CMER (AAAI 2026), and ScriptMoE (Sep 2026) ([OpenOCR README](https://github.com/Topdu/OpenOCR)).
   - OpenOCR reproduces about 30 STR methods with training and evaluation, all on U14M-Filter, which is the fairest public comparison available ([OpenOCR](https://github.com/Topdu/OpenOCR), [SVTRv2 T3](https://arxiv.org/html/2411.15858v2)).
   - Industry: PP-OCRv5 (2025) and PP-OCRv6 (June 2026) stay with CTC and argue that data matters more than architecture ([2603.24373](https://arxiv.org/abs/2603.24373)).

---

## 3. "CTC is making a comeback" (item 3)

### SVTRv2 ([arXiv](https://arxiv.org/abs/2411.15858), [HTML](https://arxiv.org/html/2411.15858v2), ICCV 2025)

SVTRv2 targets CTC's three classic weaknesses.

- **Distortion from fixed resizing.** SVTRv2 answers it with **MSR (multi-size resizing)**: pick the input size by aspect ratio R=W/H. R<1.5 → 64×64. 1.5≤R<2.5 → 48×96. 2.5≤R<3.5 → 40×112. R≥3.5 → H=32, W=⌊R⌋·32 ([SVTRv2](https://arxiv.org/html/2411.15858v2)). Wide text keeps its true width, which is the long-text win.
- **Irregular (curved or rotated) text breaks CTC's monotonic left-to-right alignment.** SVTRv2 answers it with **FRM (feature rearrangement module)**. A horizontal step runs self-attention within each row. A vertical step uses a learned selecting token over each column. The two steps turn the 2D feature map into a 1D reading-order sequence. FRM is **kept** at inference. Ablation gains on the multi-oriented subset: horizontal alone +2.03%, both steps +2.46% ([SVTRv2](https://arxiv.org/html/2411.15858v2)).
- **No linguistic context.** SVTRv2 answers it with **SGM (semantic guidance module)**. SGM predicts each character from its left and right context strings by attending over visual features, and it is **training-only**, omitted at inference with "no time cost" ([abstract](https://arxiv.org/abs/2411.15858)). Loss weights are CTC 0.1 and SGM 1. Ablation gains: Com +0.41, U14M +2.28, OST +5.11 ([SVTRv2](https://arxiv.org/html/2411.15858v2)).
- **Training recipe.** AdamW, LR 6.5e-4, batch size 1024, 20 epochs in two phases (first without SGM, then with it) ([SVTRv2](https://arxiv.org/html/2411.15858v2)).
- **Result.** At the same U14M-Filter protocol, SVTRv2-B beats MAERec, PARSeq, CPPD, LISTER and ABINet on U14M, LTB and OST, and runs at 143 FPS against 17–125 FPS for the others ([SVTRv2 T3](https://arxiv.org/html/2411.15858v2)). SVTRv2-T (5.1M) reaches Com 94.78 and U14M 79.05, ahead of ABINet's 75.72 ([SVTRv2](https://arxiv.org/html/2411.15858v2)).
- **Caveats.**
  - MSR changes the input tensor shape per aspect-ratio bucket. That is fine for dynamic-shape runtimes, but fixed-shape mobile exports (MNN, CoreML) need several exported shapes or padding.
  - FRM attention over columns adds ops that are not CNN-only.
  - FPS was measured in PyTorch dynamic mode on a 1080Ti, not on a phone ([SVTRv2](https://arxiv.org/html/2411.15858v2)).

### PP-OCR line: GTC (guided training of CTC)

- **PP-OCRv3.** An attention head (SAR) guides CTC training and is "removed entirely at prediction time". It added +1.8% (74.0 → 75.8) at unchanged 7.6 ms CPU ([PP-OCRv3](https://arxiv.org/html/2206.03001)). The cumulative ablation goes from 74.0 (SVTR-LCNet h48) to 79.4 with GTC, TextConAug, TextRotNet, U-DML and UIM ([PP-OCRv3](https://arxiv.org/html/2206.03001)).
- **PP-OCRv4.** SAR is replaced by **NRTR** (GTC-NRTR) for "more stable" guidance. The release adds DKD distillation applied to the NRTR head and length-averaged CTC logits, multi-scale height training (32/48/64), and a Lite-Neck (12M → 9.6M params). Total: 71.50 → 75.45 in-house, 8.54 → 9.8 ms (CPU, OpenVINO, 3×48×320) ([PP-OCRv4 doc](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)).
- **PP-OCRv5.** Dual branch: GTC-NRTR in training, and "only the lightweight SVTR-HGNet branch is used for prediction" ([3.0 report](https://arxiv.org/html/2507.05595v1)). The local repo config `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml` shows MultiHead [CTCHead with SVTR neck dims 120 and `use_guide: True`, NRTRHead nrtr_dim 384] with a CTC-only post-process.
- **PP-OCRv6.** CTC at inference. NRTR is an "auxiliary regularizer discarded at inference" ([PP-OCRv6](https://arxiv.org/html/2606.13108)).
- **Common thread.** SVTRv2's SGM and Paddle's GTC are the same idea: put linguistic and attention supervision on the shared visual features during training, then deploy a pure CTC graph. SGM differs in that it conditions on *both* left and right context strings, rather than an AR decoder's left-only context.

---

## 4. Lightweight backbones for recognition (item 4)

| Backbone | Used in | Key ideas | Evidence |
|---|---|---|---|
| MobileNetV1-enhance | PP-OCRv2 rec `[UNVERIFIED]` | DW-separable convs, height-strided | `[UNVERIFIED]` (memory) |
| PP-LCNet (V1) | PP-OCRv3 SVTR-LCNet | First 3 LCNet stages + 2 SVTR global blocks after pooling, h=48 | [PP-OCRv3](https://arxiv.org/html/2206.03001) |
| PP-LCNetV3 | PP-OCRv4 / v5 mobile rec | Learnable affine transform, improved reparameterization and activation, depth and width adjustments | [PP-OCRv4](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html); v5 uses scale 0.95 (local config) |
| PP-HGNetV2 (B4) | PP-OCRv5 server rec ("SVTR-HGNet") | HG blocks, GPU-friendly | [3.0 report](https://arxiv.org/html/2507.05595v1), local `PP-OCRv5_server_rec.yml` |
| LCNetV4 | PP-OCRv6 all tiers | MetaFormer (reparam DW-conv token mixer + channel mixer). Stride (2,1) in stages 3–4, then height average-pool to a 1D sequence | [PP-OCRv6](https://arxiv.org/html/2606.13108) |
| SVTR-Tiny | SVTR | 6.03M params, 4.5 ms on 1080Ti ([SVTR](https://arxiv.org/pdf/2205.00159)), but **97 ms on CPU-MKLDNN vs 8.5 ms for CRNN-based PP-OCRv2**. The mixing blocks are the bottleneck ([PP-OCRv3](https://arxiv.org/html/2206.03001)) | Lesson: transformer FLOPs ≠ CPU latency |
| RepSVTR | OpenOCR mobile rec | Reparameterized SVTR; details not on the doc page | [OpenOCR doc](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md); design `[UNVERIFIED]` |
| RepViT | OpenOCR mobile **detector** (`repvit_db.yml`) | — | [OpenOCR doc](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md). Use as a *recognition* backbone `[UNVERIFIED]` |
| FastViT, MobileNetV4, EfficientViT | — | Generic mobile backbones | I found **no primary STR paper** using them as a recognizer backbone in this session `[UNVERIFIED gap]` |

**Height and width strategy.**

- SVTR: H/4 patch embed, then two merges that halve the height with stride (2,1) and keep the width. The output is 1×W/4 ([SVTR](https://arxiv.org/pdf/2205.00159)). A width/4 sequence gives roughly 80 CTC frames for a 320-px line.
- PP-OCRv6: stride (2,1) in the late stages, then a height pool ([PP-OCRv6](https://arxiv.org/html/2606.13108)).

**32 vs 48 px height.** In PP-OCRv3, moving SVTR-LCNet from h32 to h48 raised accuracy 71.9 → 74.0 at 6.6 → 7.6 ms ([PP-OCRv3](https://arxiv.org/html/2206.03001)). Every PP-OCR version since then uses 48 for inference. v4, v5 and v6 train multi-scale at 32/48/64 ([PP-OCRv4](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html), [PP-OCRv6](https://arxiv.org/html/2606.13108)). In v4 this improved end-to-end accuracy by 0.5% but not rec-only accuracy.

**Variable width and long text.**

- CTC with a W/4 or W/8 sequence handles any width.
- AR and parallel decoders with a max length of 25 score 0.0 on LTB ([SVTRv2 T3](https://arxiv.org/html/2411.15858v2)).
- SVTRv2's MSR bucket for R≥3.5 keeps H=32 and scales the width.
- SMTR uses a sub-string matching approach for out-of-length text ([arXiv](https://arxiv.org/abs/2407.12317)).
- The PP-OCRv4 latency was measured at 3×48×320, and real variable widths change speed ([PP-OCRv4](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)).

---

## 5. Multilingual and large-vocabulary issues (item 5)

**The facts.**

- PP-OCRv5 uses one model for Simplified Chinese, Traditional Chinese, Pinyin, English and Japanese ([3.0 report](https://arxiv.org/html/2507.05595v1)).
- Its dictionary `ppocrv5_dict.txt` has **18,383 lines**. I counted this in the local repo copy (`torchocr/utils/dict/ppocrv5_dict.txt`). I did not cross-check it against upstream: `[verify against upstream PaddleOCR file]`.
- The official docs say the larger dictionary makes v5 rec slower than v4. That comes from a search snippet of the [PP-OCRv5 intro doc](https://www.paddleocr.ai/v3.0.3/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html) and is not fully re-read.
- The rest of v5's multilingual coverage ships as **separate per-script models**, each 7.5–14 MB: latin, korean, eslav, thai, greek, arabic, cyrillic, devanagari, en ([module doc](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)).
- PP-OCRv6 medium and small cover 50 languages: Chinese, Japanese, English and 46 Latin-script languages, via about 200 added diacritics. **Tiny drops Japanese** to avoid about 4,000 Kanji/Kana entries. The paper reports no other head-compression technique ([PP-OCRv6](https://arxiv.org/html/2606.13108)).

**Head-size arithmetic (my calculation, from the local config).**

- CTCHead is a single Linear from the neck dim to the class count. Classes = 18,383 + space + blank = 18,385. With neck dim 120: 120×18,385 + 18,385 ≈ **2.22M params**.
- The mobile model checkpoint is 16 MB ([module doc](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)). At FP32 that is about 4M params total, so the head is plausibly **about 50% or more of the deployed recognizer**. That share is `[derived estimate; verify by counting params]`.
- The head also produces T×18,385 logits per line (T≈80 for W=320), so softmax and argmax cost and memory scale with the vocabulary too.
- The tiny PP-OCRv6 recognizer is only 1.1M params in total ([PP-OCRv6](https://arxiv.org/html/2606.13108)). A full 18k-class head at dim 80 would be about 1.5M by itself, so tiny's dictionary must be reduced. The paper confirms a "smaller dictionary" but gives no size. My calculation is consistent with that.

**Published coping techniques.**

- **Shrink the vocabulary per deployment** (v5 per-script models; v6 tiny dropping Japanese). This is the pragmatic industrial answer ([module doc](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html), [PP-OCRv6](https://arxiv.org/html/2606.13108)).
- **Hierarchical or radical codebooks: HierCode** ([arXiv 2403.13761](https://arxiv.org/abs/2403.13761), Pattern Recognition 2025). It replaces one-hot outputs with multi-hot hierarchical binary-tree codes plus prototype learning. The authors cite one-hot heads' cost for "on-device deployment" and claim "significantly fewer parameters" plus zero-shot recognition of unseen characters. The abstract gives no numbers.
- **Radical branches as training-only regularizers** (for example SAN's auxiliary radical branch). This does not shrink the head. Source: secondary, `[UNVERIFIED]`.
- **Script-aware MoE: ScriptMoE.** A shared encoder plus a sparse decoder routed per script (top-2 experts plus a shared expert) ([arXiv](https://arxiv.org/abs/2609.24058)). It shows that the v5 recognizer, not the detector, is the multilingual bottleneck: swapping the recognizer raises CC-OCR F1 from 65.71 to 80.89. Parameter count and decoder type are not given in the abstract.
- **Low-rank or factorized heads and shared input/output embeddings.** I found **no STR-specific primary source** in this session. These are standard tricks (factorize W = U·V with rank r≪dim). They are `[UNVERIFIED for STR]` and an open idea, listed in §8.

---

## 6. End-to-end spotting on edge (item 6)

- **PGNet** (AAAI 2021) is a single-shot, fully convolutional spotter. It uses PG-CTC to decode characters gathered along a 2D center line, with no NMS or RoI, and runs at 46.7 FPS on Total-Text ([arXiv](https://arxiv.org/abs/2104.05458)). It ships in PaddleOCR `[UNVERIFIED that it is maintained in 3.x]`.
- **Context-Free TextSpotter** (2021) has 3M params and runs real-time on GPU ([arXiv 2106.05611](https://arxiv.org/abs/2106.05611)). It reportedly runs under 100 ms on the iPhone 11 Pro Neural Engine (secondary summary, `[UNVERIFIED]`).
- I found **no 2025–2026 lightweight end-to-end spotter** that displaced det+rec in my searches `[gap; not exhaustive]`.
- **Why two-stage det + rec is still the edge default.** The evidence:
  1. Every major 2025–2026 edge release (PP-OCRv5, PP-OCRv6, OpenOCR mobile) is two-stage. The PP-OCRv5 CVPR 2026 paper explicitly argues that data, not architecture, raises the ceiling of "efficient two-stage OCR pipelines" ([2603.24373](https://arxiv.org/abs/2603.24373)).
  2. The modules are separable. ScriptMoE improved CC-OCR by swapping only the recognizer ([arXiv](https://arxiv.org/abs/2609.24058)).
  3. Recognition on fixed-height crops batches well and uses height-strided CNNs. Spotters must keep high-res feature maps for small text.
  4. Detection and recognition can be quantized, distilled and versioned independently.
- Points 3 and 4 are my engineering reasoning, not cited.

---

## 7. Open problems for edge STR (item 7)

- **Low-res, blurry and handwritten text.**
  - PP-OCRv5 mobile rec scores only 0.4166 on handwritten Chinese and 0.4944 on handwritten English, against 0.86–0.88 on printed text. This is from the HF card, and the column mapping is inferred ([HF](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec)).
  - The 3.0 report says v5 is "slightly behind multimodal solutions" on handwritten English ([3.0 report](https://arxiv.org/html/2507.05595v1)).
  - On BCTR handwriting, SVTRv2-B scores 67.0 against 99.5 on documents ([SVTRv2](https://arxiv.org/html/2411.15858v2)).
- **Long lines.** BCTR long text (>25 chars) is 52.8 for SVTRv2-B and LTB is 50.2, so even the best CTC model is near 50% ([SVTRv2](https://arxiv.org/html/2411.15858v2)). Fixed-length decoders score 0.
- **Vertical and rotated text.** CTC assumes a monotonic 1D order. FRM (SVTRv2) and the MSR 64×64 bucket address near-square crops. PP-OCRv5 claims vertical-text support ([HF card](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec)), but the mechanism is not described `[UNVERIFIED]`. The PP-OCR pipeline relies on a separate text-line orientation classifier `[UNVERIFIED for v5/v6 defaults]`.
- **Confidence calibration.** CTC sequence confidence (mean or product of per-frame maxima) is uncalibrated in practice. I found no STR-edge calibration paper in this session `[gap]`.
- **Hallucination vs faithfulness.** The PP-OCRv5 paper claims "fewer hallucinations" than billion-parameter VLMs ([2603.24373](https://arxiv.org/abs/2603.24373)). PARSeq argues that external LMs can "rectify correct predictions" wrongly ([arXiv](https://arxiv.org/abs/2207.06966)).
  - CTC's per-frame alignment makes it structurally less likely to invent characters that have no visual support. This is a reasonable claim, but I found no quantitative primary source in this session, so it stays an argument.
  - Training-only LM guidance (GTC, SGM) keeps that inference-time faithfulness.
- **Benchmarks are saturated.** Six-benchmark ensembles miss only 2.91% of images, while models average 66.53% on Union14M ([arXiv](https://arxiv.org/abs/2307.08723)). Edge evaluation should use U14M-L, LTB, OST and an in-domain set, not Com.

---

## 8. Actionable ideas for a CTC-based mobile recognizer (for this repo)

1. **Keep CTC at inference and add a training-only guide head.** Options: NRTR (GTC-NRTR as in PP-OCRv4/v5/v6) or a bidirectional-context SGM as in SVTRv2. The local `PP-OCRv5_mobile_rec.yml` already has MultiHead CTC+NRTR. **Try swapping NRTR for an SGM-style left/right context head.** SVTRv2's ablation showed +2.28 on U14M and +5.11 on OST at zero inference cost ([SVTRv2](https://arxiv.org/html/2411.15858v2)).
2. **Audit the classifier head first.** Count backbone vs neck vs head params for the exported model. If the head is ≥50% (my estimate), then:
   - (a) restrict the dictionary to the deployment's scripts (the PP-OCRv6 tiny approach),
   - (b) try a low-rank factorized head (120→r→18,385, r≈32–64, `[untested idea]`), or
   - (c) evaluate HierCode-style multi-hot codes for CJK ([arXiv](https://arxiv.org/abs/2403.13761)).
3. **Use input height 48 and multi-scale training at 32/48/64.** This is proven across PP-OCRv3 through v6 ([PP-OCRv3](https://arxiv.org/html/2206.03001), [PP-OCRv6](https://arxiv.org/html/2606.13108)).
4. **Use aspect-ratio bucketing at inference** (a simplified MSR). Export 2–3 fixed shapes for MNN or CoreML, for example 48×160, 48×320 and 48×640. Avoid squashing long lines into 320, which is where CTC beats AR on LTB ([SVTRv2](https://arxiv.org/html/2411.15858v2)).
5. **Prefer reparameterized CNN token mixers over self-attention in the backbone** (LCNetV3/V4, RepSVTR). Keep at most 1–2 attention blocks in the neck, at W/4 resolution. SVTR-Tiny's attention mixing was 11× slower on CPU than a CNN-based CRNN pipeline ([PP-OCRv3](https://arxiv.org/html/2206.03001)).
6. **Distill from a large teacher that shares the same dictionary.** PP-OCRv6 tiny distils from medium ([PP-OCRv6](https://arxiv.org/html/2606.13108)). PP-OCRv4 used DKD on the NRTR head plus CTC logits ([PP-OCRv4](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)). An SVTRv2-B (ICCV 2025, code in OpenOCR) teacher is a strong public option.
7. **Clean the data.** PP-OCRv4's two-model confidence filter (drop >0.95 from a weak model, drop <0.15 from a strong one) cut training from 2 weeks to 5 days and *added* +1.2% ([PP-OCRv4](http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)). The PP-OCRv5 CVPR paper is entirely about data difficulty, label accuracy and diversity ([2603.24373](https://arxiv.org/abs/2603.24373)).
8. **Evaluate on U14M-L, LTB and OST plus in-domain data**, measured as on-device latency at the real width distribution, not at 320.
9. **Calibrate confidence** with temperature scaling on CTC frame posteriors against a held-out in-domain set. This is an `[untested idea]`, motivated by the gap noted in §7.

---

## 9. Sources

Primary (fetched this session):
- SVTR (IJCAI 2022): https://arxiv.org/abs/2205.00159 , PDF tables 1 and 4: https://arxiv.org/pdf/2205.00159
- SVTRv2 (ICCV 2025): https://arxiv.org/abs/2411.15858 , https://arxiv.org/html/2411.15858v2
- PP-OCRv3: https://arxiv.org/abs/2206.03001 , https://arxiv.org/html/2206.03001
- PP-OCRv4 (official doc): http://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html
- PaddleOCR 3.0 Technical Report (PP-OCRv5): https://arxiv.org/abs/2507.05595 , https://arxiv.org/html/2507.05595v1
- PP-OCRv5 specialized 5M model (CVPR 2026 listing): https://arxiv.org/abs/2603.24373 , https://cvpr.thecvf.com/virtual/2026/poster/38525
- PaddleOCR text recognition module doc (v4/v5/v6 tables): http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html
- PP-OCRv6 paper: https://arxiv.org/html/2606.13108 ; doc: https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html
- PP-OCRv5 mobile rec HF card: https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec
- OpenOCR repo: https://github.com/Topdu/OpenOCR ; OpenOCR system doc: https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md
- PARSeq (ECCV 2022): https://arxiv.org/abs/2207.06966
- CPPD: https://arxiv.org/abs/2307.12270
- Union14M / MAERec (ICCV 2023): https://arxiv.org/abs/2307.08723
- CLIP4STR (T-IP): https://arxiv.org/abs/2305.14014
- DTrOCR (WACV 2024): https://arxiv.org/abs/2308.15996
- TrOCR: https://arxiv.org/abs/2109.10282
- IGTR (TPAMI): https://arxiv.org/abs/2401.17851
- UniRec-0.1B: https://arxiv.org/abs/2512.21095
- ScriptMoE: https://arxiv.org/abs/2609.24058
- HierCode (Pattern Recognition 2025): https://arxiv.org/abs/2403.13761
- PGNet (AAAI 2021): https://arxiv.org/abs/2104.05458

Seen via search results only (abstract-level, not fully read):
- SMTR / LTB: https://arxiv.org/abs/2407.12317
- LISTER (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.html
- MDiff4STR (AAAI 2026): https://ojs.aaai.org/index.php/AAAI/article/view/37370 , https://arxiv.org/html/2512.01422v1
- Context-Free TextSpotter: https://arxiv.org/abs/2106.05611
- PP-OCRv5 intro doc (dictionary and speed remark): https://www.paddleocr.ai/v3.0.3/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html

Local repo evidence (not web sources):
- `torchocr/utils/dict/ppocrv5_dict.txt` (18,383 lines)
- `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml` (PPLCNetV3 ×0.95, SVTR neck dims 120, CTC+NRTR MultiHead, 3×48×320, multi-scale 32/48/64)
- `configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml` (PPHGNetV2_B4)
