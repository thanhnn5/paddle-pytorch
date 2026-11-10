# 01 — PaddleOCR (Baidu) lineage: PP-OCR v1 → v6, PaddleOCR 3.x, PP-StructureV3, PP-ChatOCRv4, PaddleOCR-VL

*Survey note, compiled 2026-11-10. Every number carries a source tag such as [S3], and the tags resolve in the **Sources** list at the end. Items not confirmed from a primary source are tagged `[UNVERIFIED]`. "Config" means the official YAML in `PaddlePaddle/PaddleOCR@main` [S12].*

> **How to read benchmark numbers.** Every PP-OCR generation reports accuracy on a **different in-house Baidu test set**, and the sets grew harder over time. Cross-version numbers are therefore only comparable *within one table from one source*. Examples: the PP-OCRv4 doc gives PP-OCRv3's end-to-end (e2e) Hmean as 57.99%, while the PP-OCRv3 paper gives 62.9%. The PP-OCRv6 table states it uses a different eval set from the v5/v4 numbers [S4][S3][S7].

---

## 1. Version-by-version comparison

| | **PP-OCR (v1)** | **PP-OCRv2** | **PP-OCRv3** | **PP-OCRv4** | **PP-OCRv5** | **PP-OCRv6** |
|---|---|---|---|---|---|---|
| Date / vehicle | arXiv Sep 2020 [S1] | arXiv Sep 2021 [S2] | arXiv Jun 2022 [S3] | PaddleOCR 2.7 docs, 2023 (no paper; date `[UNVERIFIED]`) [S4] | PaddleOCR 3.0 tech report, Jul 2025 [S5] | PaddleOCR v3.7.0, 2026-06-11 (no arXiv paper found) [S9][S8] |
| Det. algorithm | DB [S1] | DB [S2] | DB [S3] | DB [S4] | DB [S5][S12] | DB [S12] |
| Det. backbone (mobile) | MobileNetV3-large ×0.5, SE removed [S1] | same as v1 [S2] | MobileNetV3 [S3] | PP-LCNetV3 (×0.75 in config) [S4][S12] | PP-LCNetV3 ×0.75 [S5][S12] | PPLCNetV4 tiny/small [S8][S12] |
| Det. backbone (server) | ResNet18_vd ("large-scale" baseline) [S1] | (teacher ResNet18) [S2] | (teacher ResNet50) [S3] | PPHGNet_small [S12] | PPHGNetV2_B4 [S5][S12] | PPLCNetV4 medium [S12] |
| Det. neck | FPN, 96 inner channels [S1] | same [S2] | **RSE-FPN** (student); **LK-PAN** (teacher) [S3] | RSEFPN (mobile); LKPAN (server) [S12] | RSEFPN (mobile); LKPAN (server) [S12] | **RepLKFPN** (tiny/small); **RepLKPAN** (medium, per config) [S8][S12] |
| Det. head | DB head [S1] | DB head [S2] | DB head [S3] | **PFHead** [S4] | PFHead retained [S5] | DBHead + aux deep supervision at P2/P3/P4 [S8][S12] |
| Det. training tricks | cosine LR, warm-up, FPGM pruning [S1] | **CML** (2 students + teacher), **CopyPaste** [S2] | CML with better teacher (LK-PAN+DML) [S3] | **DSR** (shrink ratio 0.4→0.6), CML + KL [S4] | Distillation from **GOT-OCR2.0 visual encoder** teacher, hard-case mining via **ERNIE-4.5-VL-424B-A47B** [S5] | DiceFocal loss, aux losses; train size 640×640 [S8][S12] |
| Rec. algorithm | CRNN (BiLSTM + CTC) [S1] | CRNN + CTC [S2] | **SVTR-LCNet** + CTC, GTC attention guide [S3] | SVTR_LCNetV3 + CTC, **GTC-NRTR** guide [S4] | SVTR-LCNet / SVTR-HGNet + CTC, GTC-NRTR [S5] | PPLCNetV4 + **LightSVTR** + CTC, NRTR aux [S8][S12] |
| Rec. backbone | MobileNetV3-small ×0.5 [S1] | **PP-LCNet** [S2] | PP-LCNet (first 3 stages) + 2 SVTR Global Mix blocks [S3] | **PP-LCNetV3** (mobile); PPHGNet_small (server) [S4][S12] | PP-LCNetV3 ×0.95 (mobile); **PPHGNetV2_B4** (server) [S5][S12] | **PPLCNetV4** (MetaFormer-style, re-param DW conv) [S8] |
| Rec. input | 3×32×320 [S1] | 3×32×320 `[UNVERIFIED]` | 3×48×320 [S3] | multi-scale H∈{32,48,64}, W=320; eval 48×320 [S4][S12] | same multi-scale [S12] | same multi-scale [S12] |
| Dictionary | 6,622 chars [S1] | 6,622 (same `ppocr_keys_v1`) `[UNVERIFIED]` | 6,622 (`ppocr_keys_v1`) [S12] | `ppocr_keys_v1` (mobile/server); server_rec_doc 15,000+ chars [S12][S7] | `ppocrv5_dict.txt`, 18,383 lines [S15] | `ppocrv6_dict.txt` 18,708 lines; tiny 6,904 lines [S15] |
| Total system size | 3.5 MB (Chinese), 2.8 MB (alphanumeric) [S1] | 11.6 MB [S2] | 15.6 MB [S3] | 15.8 MB [S4] | server det 84.3 MB / rec 81 MB; mobile det 4.7 MB / rec 16 MB [S7] | medium det 59.4 / rec 73.3 MB; small 9.6 / 20.4 MB; tiny 1.9 / 4.4 MB [S7]; params 1.5M (tiny) / 7.7M (small) / 34.5M (medium) [S9] |
| Headline gain | — | e2e Hmean +7.3 pts over PP-OCR mobile at the same cost [S2] | e2e Hmean +5.3 pts over v2 [S3] | e2e Hmean 57.99→62.24% vs v3 [S4] | "13 percentage point" e2e gain over v4 (internal sets) [S6] | medium: det +4.6, rec +5.1 over v5_server [S8][S9] |

---

## 2. Per-version detail

### 2.1 PP-OCR (v1), 2020 [S1]

**Pipeline:** DB detector → box rectification + text **direction classifier** → CRNN recognizer. The paper calls this "19 effective strategies" [S2].

**Detection (6 strategies)** [S1]
- Light backbone: **MobileNetV3-large ×0.5**, chosen from PaddleClas benchmarks.
- Light head: FPN inner channels cut from 256 to **96**. Model size went 7 MB→4.1 MB and CPU time 406→213 ms.
- **SE removed**, because SE helps little at 640×640 input and costs time. Size 4.1→2.5 MB "but the accuracy has no effect" (paper text); CPU time 213→173 ms (table).
- Cosine LR decay and LR warm-up.
- **FPGM pruning:** size 2.6→1.4 MB and Snapdragon 855 (SD855) time 164→133 ms, with Hmean 0.6239→0.6169.
- Final detector is **1.4 MB**.

**Direction classifier (4 strategies)** [S1]
- MobileNetV3-small ×0.35.
- Augmentation: BDA + RandAugment.
- Input 48×192 instead of 32×100.
- **PACT quantization**, modified to handle hard-swish. Size 0.85→0.46 MB, SD855 time 3.21→2.38 ms, accuracy 0.9403→0.9456. The classifier is about 500 KB.

**Recognition (9 strategies)** [S1]
- Backbone: MobileNetV3-small ×0.5.
- Augmentation: BDA + **TIA**.
- Cosine LR decay.
- Feature-map resolution: input 32×320, with the second downsampling stride changed from (2,1) to **(1,1)**, worth +5.27% accuracy.
- L2 decay 1e-5, worth +3.4%.
- Warm-up.
- Light head: sequence feature dimension **48**. Going from 256 to 48 channels shrank the model from 23 MB to 4.6 MB with no accuracy loss.
- Pre-trained model: accuracy 65.81%→69%.
- PACT quantization, with the LSTM layers left unquantized: 4.6 MB→1.5 MB, SD855 12→11 ms, accuracy 0.6581→0.674.
- Final recognizer is **1.6 MB** (Chinese+English) or **0.9 MB** (alphanumeric).

**Losses:**
- Detection uses the DB loss.
- Recognition uses CTC.
- The PACT clipping function was modified for hard-swish activations, and an L2 coefficient of 0.001 was put on the PACT parameters [S1].

**Data** [S1]
- Detection: 97K images (68K real, 29K synthetic).
- Classifier: 600K images.
- Recognition: **17.9M** images (1.9M real, 16M synthetic).
- System eval set: 300 images.

**System results** [S1]

| System | F-score | Size | CPU (Xeon 6148) | T4 GPU |
|---|---|---|---|---|
| Ultra-light | 0.5193 | 8.1 MB | 421 ms | 137 ms |
| Server (Res18_vd det + Res34_vd rec) | 0.5414 | 155.1 MB | 1199 ms | 204 ms |

Slimming (FPGM + PACT) cut the system from 8.1 MB to **3.5 MB** with no F-score loss (0.5193→0.5210). SD855 time went 306→268 ms.

**Why:** an ultra-light OCR system for CPU and embedded devices, built from a bag of tricks rather than a new architecture [S1].

### 2.2 PP-OCRv2, 2021 [S2]

**What changed:** 5 new tricks, with "precision" (actually e2e Hmean) 7% higher than PP-OCR at the same inference cost [S2].

**Detection** [S2]
- **CML (Collaborative Mutual Learning).** Two MobileNetV3-large ×0.5 students learn from each other (DML) plus a frozen **ResNet18** DB teacher.
- Loss = L_gt + L_dml + L_distill.
  - L_gt is DB's loss: BCE on the probability map + α·Dice on the binary map + β·L1 on the threshold map, with α=5 and β=10.
  - L_dml is a symmetric KL between the two students' probability maps.
  - L_distill = γ·BCE + Dice against the teacher maps, which are **dilated** with a 2×2 kernel. γ=5.
- **CopyPaste** augmentation for text instances.
- Ablation (Hmean): 0.759 (PP-OCR det) → DML 0.777 → CML 0.789 → +CopyPaste **0.795**. Model stays at 3.0 MB and 129 ms CPU.
- **Why:** plain distillation gains little when teacher and student are close in accuracy or very different in structure [S2].

**Recognition** [S2]
- **PP-LCNet** backbone. It is MobileNetV1-based for MKLDNN friendliness on Intel CPUs, with:
  - h-swish activations,
  - SE only near the tail,
  - 5×5 kernels at the tail,
  - a 1280-d 1×1 conv after GAP.
- Accuracy 0.667→0.693, size 5.0→8.0 MB, CPU 7.7→6.2 ms.
- **U-DML:** two same-architecture networks trained from scratch.
  - Loss = CTC(S) + CTC(T) + L_dml + L_feat.
  - L_dml is symmetric KL on the softmax outputs.
  - L_feat is an MSE (L2) between the backbone feature maps.
  - Training runs 800 epochs with piece-wise LR.
  - The CTC head was expanded to 2 FC layers, worth about +1.5%.
  - Overall U-DML gain: +4.6% accuracy (0.693→0.739).
- **Enhanced CTC loss** = L_ctc + λ·L_center.
  - L_center = Σ_t ‖x_t − c_{y_t}‖², with y_t taken from greedy CTC argmax decoding.
  - λ = 0.05.
  - Gain: +0.9% (→0.748).
  - **Why:** to separate visually similar Chinese characters.

**System** [S2]

| System | Hmean | Size | CPU | T4 |
|---|---|---|---|---|
| PP-OCRv2 | **0.576** | 11.6 MB | 330 ms | 111 ms |
| PP-OCR mobile | 0.503 | 8.1 MB | 356 ms | 116 ms |
| PP-OCR server | 0.570 | 155.1 MB | 1056 ms | 200 ms |

The paper planned slim versions (PACT, FPGM) for a future "PP-OCRv2-tiny" but did not validate them in the paper [S2].

### 2.3 PP-OCRv3, 2022 [S3]

**What changed:** 9 upgrades (3 for detection, 6 for recognition). Recognition moved from CRNN to **SVTR-based** models (no RNN). e2e Hmean is +5.3 pts over v2 at the same CPU cost and 22% faster on T4 [S3].

**Detection** [S3]
- Still CML. The upgrades are a better teacher and a better student.
- **LK-PAN** (teacher): PAN path-augmentation convolutions grow from 3×3 to **9×9** to help with large fonts and extreme aspect ratios. DB-R50 Hmean 83.5→85.0%.
- **DML** on the teacher: 85.0→**86.0%**.
- **RSE-FPN** (student): FPN convolutions are replaced by residual-SE conv. The residual exists because plain SE suppressed channels in the thin FPN. Hmean 81.3→84.5% with about 6% more time.
- Final student DB-MV3-RSE-FPN-CML: **85.4%**, 3.6 MB, 124 ms. The PP-OCRv2 detector scores 83.2% on the same set.

**Recognition (6 strategies)** [S3]

1. **SVTR-LCNet.** SVTR-Tiny reaches 80.1% but is about 10× slower on MKLDNN CPU (97 ms). It was cut down in steps:
   - replace the first half with PP-LCNet stages 1–3,
   - reduce to 2 Global Mix blocks,
   - move the blocks after pooling,
   - raise input height from 32 to 48.

   Result: 74.0%, 12 MB, 7.6 ms.
2. **GTC:** an attention branch guides CTC during training and is removed at inference. +1.82%. The PP-OCRv4 doc states this v3 guide branch was **SAR** [S4].
3. **TextConAug:** concatenates two images from a batch, adapted from ConCLR. +0.5%.
4. **TextRotNet:** self-supervised pretraining on unlabeled text lines (STR-Fewer-Labels style), used to initialize weights. +0.6%.
5. **U-DML:** supervises the PP-LCNet features, the SVTR output and the attention output together. +1.5%.
6. **UIM (Unlabeled Images Mining):** a high-precision recognizer pseudo-labels unlabeled images, and only high-confidence samples are kept. +1.0%, for a final accuracy of **79.4%**.

**Data** [S3]
- Detection: 127K images.
- Recognition: 18.5M, of which 7M are real. The real set includes **810K images mined from unlabeled LSVT via UIM**, about 3M PubTabNet crops and 750K scraped financial-report images.
- e2e eval set: 800 images.

**System:** v3 reaches Hmean **62.9%**, 15.6 MB, 331 ms CPU, 87 ms T4. v2 scores 57.6% on the same set [S3]. Docs report v3 English e2e Hmean 53.14% versus 42.17% for PP-OCR_en [S13].

**Deployment:** the Paddle-Lite Android demo ships PP-OCRv3 at 16.2 MB as .nb files, and a **slim (INT8) version at 5.9 MB** [S14]. Mobile latency is not stated there; SD-series latency for v3 is `[UNVERIFIED]`.

### 2.4 PP-OCRv4, 2023 (docs only, no arXiv paper) [S4]

**Detection (mobile)** [S4]

| Step | Hmean | Size | CPU (OpenVINO) |
|---|---|---|---|
| PP-OCRv3 student baseline | 76.22% | 3.4 MB | 69 ms |
| + **PFHead** (parallel fusion of upsample and transposed-conv branches, then 1×1 conv) | 76.97% | — | 96 ms |
| + **DSR** (Dynamic Shrink Ratio, linear 0.4→0.6 over training) | 78.24% | — | — |
| + PP-LCNetV3 backbone | 79.08% | 4.8 MB | — |
| + CML with a KL loss between student and teacher response maps | **79.87%** | — | 67 ms |

The page's own text gives 79.56% for the final CML step, which contradicts its table.

**Server detection (from configs):** PPHGNet_small + LKPAN(256) + PFHeadLocal [S12]. The public `PP-OCRv4_mobile_det.yml` uses a plain `DBHead` with `RSEFPN(96)` [S12]. This suggests PFHead/CML lived in distillation configs and not in the released student config `[UNVERIFIED interpretation]`.

**Recognition (mobile)** [S4]

| Step | Accuracy |
|---|---|
| PP-OCRv3 | 71.50% |
| + **DF data filtering** | 72.70% |
| + Lite-Neck + GTC | 73.21% |
| + PP-LCNetV3 | 74.18% |
| + multi-scale | 74.20% |
| + TextConAug | 74.72% |
| + U-DML/DKD | **75.45%** (table says UDML, text says DKD) |

What each step does:
- **DF (data filtering):** a weak model drops "easy" samples with confidence above 0.95, and the v3 model drops likely-mislabeled samples with confidence below 0.15. The training set shrank from tens of millions to millions, and training time from about 2 weeks to 5 days.
- **PP-LCNetV3 / SVTR_LCNetV3:** learnable affine transforms, improved re-parameterization and activation, and adjusted depth and width.
- **Lite-Neck:** 12 MB→8.5 MB at the same accuracy. Raising the CTC neck dimension from 64 to 120 brings it to 9.6 MB.
- **GTC-NRTR:** the NRTR Transformer replaces SAR as the GTC guide branch.
- **Multi-scale:** input height randomly 32/48/64. No change in rec accuracy, +0.5% e2e.
- **DKD distillation:**
  - The NRTR head uses DKD loss + CE without label smoothing.
  - The CTC head's logits are averaged along the length dimension so the task can be treated as multi-label classification.

**Losses (config):** `MultiLoss[CTCLoss, NRTRLoss]` [S12].

**Server recognition:** SVTR_HGNet with PPHGNet_small [S12]. `PP-OCRv4_server_rec_doc` adds document data and supports 15,000+ characters [S7].

**Results** [S4]
- e2e (Chinese): Hmean 57.99%→**62.24%**, 15.6→15.8 MB, 78→76 ms CPU.
- English rec: 64.04%→**70.1%**.
- Multilingual: Latin 80.00%, Arabic 75.48%, Japanese 56.50%, Korean 83.25%.

### 2.5 PP-OCRv5 (PaddleOCR 3.0), 2025 [S5][S6][S7]

**Detection** [S5]
- Backbone moves from PP-HGNet to **PP-HGNetV2** (server).
- **Feature-alignment distillation from the GOT-OCR2.0 visual encoder** as teacher. This is the first use of a large-VLM encoder as teacher in the det line.
- Data augmentation:
  - **hard-case mining by comparing against ERNIE-4.5-VL-424B-A47B**,
  - text-line multilingual synthesis, rotation, blurring and geometric transforms.
- **PFHead and DSR are retained** from v4.
- Configs: server = PPHGNetV2_B4 + LKPAN(256) + PFHeadLocal. Mobile = PPLCNetV3 ×0.75 + RSEFPN(96) + DBHead. Both use CopyPaste and 640×640 crops, with DBLoss α=5, β=10, OHEM 3 [S12].

**Recognition** [S5]
- Dual branch.
  - **GTC-NRTR** is an attention branch used only in training.
  - **SVTR-HGNet** (server) or SVTR-LCNet (mobile) uses a CTC loss and is the only branch kept at inference.
  - Figure 3 also lists "Lite-Neck" and "Multi-Scale".
- Config: `MultiLoss[CTCLoss, NRTRLoss]`, an SVTR neck with dims=120 and depth=2, NRTR dim 384, multi-scale heights 32/48/64 [S12].
- **Unified multilingual model:** Simplified Chinese, Traditional Chinese, Pinyin, English and Japanese in one model, "under 100 MB" [S5].
- Dictionary file has 18,383 lines [S15]. Docs blame the larger dictionary for the slower inference compared with v4 [S6].

**VLM/LLM-generated labels (key new ingredient)** [S5]
- ERNIE-4.5-VL-424B-A47B and "traditional models" auto-annotate and filter high-quality handwritten samples. Rare characters are synthesized.
- Large-scale labels are parsed from PDFs and e-books, with edit-distance filtering.
- The PP-OCRv5 figure lists "Label Data Using ERNIE 4.5" and "Pretrain Distillation".

**Results**
- Docs (Average / Weighted Average) [S6]:
  - Detection: server 0.827 vs v4 0.662; mobile 0.770 vs v4 0.624.
  - Recognition: server 0.8401 vs v4 0.5735; mobile 0.8015 vs v4 0.5301.
- Model-list table on a different set [S7]:
  - Detection Hmean: v5 server 83.8 / mobile 79.0; v4 69.2 / 63.8.
  - Recognition: v5 server 86.38 / mobile 81.29.
- Handwriting error rate is down 26% [S5].
- The report claims v5 (described as "0.07 B parameters") ranks first on average 1-EditDist across 17 scenarios. The comparison set includes GOT-OCR2.0, Qwen2.5-VL-72B, InternVL3-78B, Gemini 2.5 Pro and GPT-4o [S5]. These are vendor in-house numbers.

**Latency** [S6][S7]
- Pipeline on V100: 0.62 s/img (v5 mobile) vs 0.29 s/img (v4 mobile), so v5 is slower. CPU: 1.75 vs 1.37 s [S6].
- Per-model on T4 (excludes pre/post-processing), standard / HPI [S7]:
  - v5_mobile_det: 10.67 / 6.36 ms
  - v5_mobile_rec: 5.43 / 1.46 ms
  - v5_server_det CPU: 383 ms
- High-performance inference (HPI) on T4 cuts v5_mobile_rec latency by 73.1% and v5_mobile_det by 40.4% [S5].

**Multilingual expansion**
- v3.2.0 added English, Thai and Greek models [S9].
- v3.3.0 added **109-language** v5 multilingual rec models with "only 2M parameters" [S9].
- Per-script mobile rec models are 7.5–14 MB, e.g. latin 84.7%, korean 88.0% [S7].

### 2.6 PP-OCRv6 (PaddleOCR 3.7.0, 2026-06-11) [S8][S9][S12]

- **No arXiv paper found.** The primary sources are the algorithm doc [S8], the release notes [S9] and the configs [S12].
- **Three tiers** sharing one backbone family:

  | Tier | Target | Params | Det / rec size |
  |---|---|---|---|
  | tiny | edge/IoT | 1.5M | 1.9 / 4.4 MB |
  | small | mobile/desktop | 7.7M | 9.6 / 20.4 MB |
  | medium | server | 34.5M | 59.4 / 73.3 MB |

  Sources: tiers and params [S9][S8]; sizes [S7]. The model table describes tiny_det alone as "0.43M params" [S7]; the 1.5M figure therefore probably counts det + rec `[UNVERIFIED]`.
- **Languages:** 50 for small/medium, 49 for tiny (no Japanese), in one model [S8]. Dictionary files have 18,708 lines (small/medium) and 6,904 lines (tiny) [S15]. Docs add about 200 diacritics [S8].

**Backbone PPLCNetV4** [S8]
- MetaFormer-style block:
  - token mixer = 3×3 DW conv with optional SE;
  - channel mixer = 2× expand → GELU → compress + residual, with zero-init BN on compress.
- **RepDWConv** structural re-parameterization (3×3 + 1×1 + identity).
- **Task-adaptive downsampling:**
  - det uses stride-2 everywhere, giving strides 4/8/16/32;
  - rec uses (2,1) strides at stages 3–4, then height average-pooling.

**Detection** [S8][S12]
- Neck:
  - **RepLKFPN** (DilatedReparamBlock: 7×7 DW plus dilated branches). It has 118K parameters vs 172K for v5 RSEFPN (−31%), and the receptive field grows from 3×3 to 7×7 [S8].
  - Official configs: tiny = RepLKFPN(64, kernel 5); small = RepLKFPN(96, kernel 7); **medium = RepLKPAN(256)** [S12]. The doc's "RepLKFPN" wording therefore covers only tiny and small.
- Head: DBHead + **aux deep supervision at P2/P3/P4**, with weights 0.4 / 0.3 / 0.2.
- **Loss:**
  - DBLoss with `main_loss_type: DiceFocalLoss`, focal α=0.25, γ=2.5, α=5, β=10 [S12];
  - the doc calls it "DiceBCE (Dice + Focal)" [S8].
  - This is the first change to the det loss in the lineage (v1–v5 used BCE+Dice+L1 DBLoss).
- Train crop 640×640 [S12].
- No teacher/CML is visible in the released det configs `[UNVERIFIED whether distillation was used for det]`.

**Recognition** [S8][S12]
- **LightSVTR** neck: a 1×7 DW conv for local context + 1–2 Transformer layers, with *additive* skips replacing v5's concatenation.
- Configs:
  - medium: dims 192, depth 2, mlp 4, NRTR dim 512;
  - small: dims 120, mlp 2;
  - **tiny: no neck** ("reshape" + FC), **distilled from medium** [S8][S12].
- CTC at inference. NRTR head for training only. Loss `MultiLoss[CTCLoss, NRTRLoss]`. RecConAug (TextConAug) is still used. Multi-scale heights 32/48/64 [S12].

**Results** [S8]

All are in-house sets, **not comparable** with v5's published tables.

| Model | Det Hmean avg (16 categories) | Rec W-Avg (15 categories) |
|---|---|---|
| PP-OCRv6 medium | **86.2** | **83.2** |
| PP-OCRv6 small | 84.1 | 81.3 |
| PP-OCRv6 tiny | 80.6 | 73.5 |
| PP-OCRv5 server | 81.6 | 78.1 |
| PP-OCRv5 mobile | 75.2 | 73.7 |

VLM baselines on det: Gemini-3.1-Pro 46.8, GPT-5.5 45.6. On rec: Qwen3-VL-235B 74.9 [S8].

**Latency (s/img, full pipeline incl. I/O)** [S8]

| Hardware / backend | medium | small | tiny | v5 server | v5 mobile |
|---|---|---|---|---|---|
| A100 / Paddle | 0.29 | 0.25 | 0.13 | 0.32 | 0.25 |
| Xeon 8350C / OpenVINO | 1.40 | 0.59 | 0.20 | 7.30 | 0.78 |
| Apple M4 / ONNX Runtime | 5.55 | 1.29 | **0.35** | 7.20 | 1.10 |

The doc also claims a "2.37×" GPU speedup for medium, which is not reconciled with its own table (0.29 vs 0.32 s) `[UNVERIFIED]`.

**Data:** the doc gives no training-data or pseudo-label description. Whether VLM labels were used for v6 is `[UNVERIFIED]`.

**Deployment:** Paddle, ONNX Runtime, OpenVINO, TensorRT and HF Transformers (>=5.8.0) [S8]. Paddle-Lite and INT8 for v6 are not mentioned `[UNVERIFIED]`.

**Why (stated):** a single backbone serving both det and rec across edge-to-server tiers; MetaFormer-style capacity at low FLOPs [S8].

### 2.7 PaddleOCR 3.x platform, PP-StructureV3, PP-ChatOCRv4 [S5][S9]

**PaddleOCR 3.0 (2025)** [S5]
- The inference library was rebuilt on **PaddleX 3.0**: Interface, Wrapper and Foundation layers.
- High-performance inference (`enable_hpi`) auto-selects among Paddle Inference, OpenVINO, ONNX Runtime and TensorRT, with on-demand Paddle→ONNX conversion and FP16.
- Serving: FastAPI basic serving, or Triton for high-stability serving.
- On-device: **Paddle-Lite**.
- An MCP server.

**Later 3.x releases** [S9]
- v3.5.0: HF Transformers backend for 20 models, and **PaddleOCR.js** browser SDK for PP-OCRv5.
- v3.4.1: llama.cpp server backend for the VL model, plus AMD and Intel Arc support.

**PP-StructureV3** [S5]
- Pipeline: preprocessing (PP-LCNet orientation classifier + UVDoc unwarping) → PP-OCRv5 → layout analysis (**PP-DocLayout-plus** + region detection) → item recognition → reading-order post-processing.
- Item recognition uses PP-TableMagic, PP-FormulaNet_plus (2560 tokens), the **PP-Chart2Table** small VLM and PP-OCRv4_seal.
- Reading order uses an improved X-Y Cut.
- OmniDocBench overall edit distance: **0.145 EN / 0.206 ZH**. This beats MinerU-1.3.11 (0.166 / 0.310) and Gemini 2.5 Pro (0.148 / 0.212) in the report's table.

**PP-ChatOCRv4** [S5]
- Components: PP-Structure + vector retrieval (RAG) + LLM (**ERNIE-4.5-300B-A47B**) + the doc-VLM **PP-DocBee2 (3B)**, with result fusion.
- Recall@1 on 638 docs / 1,196 QA pairs: **85.55%**. Comparisons: Qwen2.5-VL-72B 80.26%, PP-ChatOCRv3 70.08%, GPT-4o 63.47%.

### 2.8 PaddleOCR-VL family (Oct 2025 →) [S10][S11][S9]

**PaddleOCR-VL (0.9B)** [S10]

Two stages:
1. **PP-DocLayoutV2** for layout and reading order.
   - It is RT-DETR + a pointer network.
   - RT-DETR is initialized from PP-DocLayout_Plus-L and trained on over 20K samples.
   - The pointer network is trained with Generalized Cross-Entropy, with RT-DETR frozen.
2. **PaddleOCR-VL-0.9B** reads each element crop.
   - Vision encoder: a NaViT-style dynamic-resolution encoder initialized from **Keye-VL**.
   - Projector: 2-layer MLP, merge size 2.
   - Language model: **ERNIE-4.5-0.3B** with 3D-RoPE.
   - Training:
     - Stage 1: 29M image-text pairs, max 1280×28×28.
     - Stage 2: 2.7M instruction samples covering OCR, table, formula and chart, max 2048×28×28.
   - Over 30M training samples were collected in total.

**Labeling** [S10]
- PP-StructureV3 produces pseudo-labels.
- **ERNIE-4.5-VL and Qwen2.5-VL** refine them via prompts.
- Hallucination filtering follows.

**Results** [S10]
- OmniDocBench v1.5 overall **92.86**, vs MinerU2.5-1.2B at 90.67.
- 109 languages.
- Throughput on A100 with FastDeploy: **1.6184 pages/s** and 2486 tokens/s, i.e. 53.1% more pages/s than MinerU2.5.

**Successors** [S11][S9]
- **PaddleOCR-VL-1.5** (arXiv 2601.21957, Jan 2026): 94.5% on OmniDocBench v1.5. Adds seal recognition and text spotting, and introduces the Real5-OmniDocBench robustness set [S11].
  - Upgrade to PP-DocLayoutV3 is reported by a search summary only `[UNVERIFIED]`.
  - Irregular-box output and cross-page table merging come from the release notes [S9].
- **PaddleOCR-VL-1.6** (v3.6.0, 2026-05-28): **96.33%** on OmniDocBench v1.6, same architecture as 1.5 [S9]. No paper found.

---

## 3. Losses across the lineage (summary)

| Stage | Detection loss | Recognition loss |
|---|---|---|
| v1 | DB: BCE(prob) + Dice(binary) + L1(threshold) [S1][S2] | CTC [S1] |
| v2 | DB loss + DML symmetric KL + distill (BCE + Dice vs dilated teacher maps) [S2] | CTC×2 + DML KL + feature MSE (U-DML); Enhanced CTC = CTC + 0.05·CenterLoss [S2] |
| v3 | Same CML family [S3] | CTC + attention (SAR) GTC branch; U-DML across SVTR, attention and features [S3][S4] |
| v4 | DB loss + CML with KL between response maps; DSR shrink schedule [S4] | CTC + NRTR (GTC); DKD + CE distillation on NRTR, length-averaged CTC logits [S4][S12] |
| v5 | DB loss (α5, β10, OHEM 3) + feature alignment to GOT-OCR2.0 encoder [S5][S12] | CTC + NRTR [S12] |
| v6 | DB with **Dice + Focal** main loss + aux P2/P3/P4 deep-supervision losses [S8][S12] | CTC + NRTR; tiny trained with KD from medium (KD loss form not stated) [S8][S12] |

## 4. Unlabeled data, pretraining, teachers, VLM labels

| Gen | Teachers / pretraining | Unlabeled / auto-labeled data |
|---|---|---|
| v1 | ImageNet / large-synthetic pretraining [S1] | 16M synthetic rec images [S1] |
| v2 | ResNet18 DB teacher (CML); teacher-free U-DML for rec [S2] | — |
| v3 | ResNet50 + LK-PAN + DML teacher [S3] | **TextRotNet** self-supervised pretraining on unlabeled lines; **UIM** (810K pseudo-labeled LSVT images) [S3] |
| v4 | DKD distillation in rec [S4] | **DF** (confidence-based filtering of training data, not pseudo-labelling) [S4] |
| v5 | **GOT-OCR2.0 visual encoder** as det teacher [S5] | **ERNIE-4.5-VL-424B-A47B** used for handwriting auto-annotation and filtering, and for det hard-case mining; PDF/e-book parsing with edit-distance filtering [S5] |
| v6 | medium → tiny KD [S8] | not disclosed `[UNVERIFIED]` |
| PaddleOCR-VL | Keye-VL encoder + ERNIE-4.5-0.3B init [S10] | PP-StructureV3 pseudo-labels refined by ERNIE-4.5-VL and Qwen2.5-VL, then hallucination-filtered [S10] |

The teacher has scaled steadily: a larger CNN of the same family (v2–v3), then a VLM encoder (v5); for data, from self-mined confident pseudo-labels (v3), to VLM labels (v5), to VLM-refined labels for the VLM itself (VL).

## 5. Deployment story

**v1** [S1]
- PaddleSlim **FPGM pruning** + **PACT** QAT, with PACT adapted for hard-swish and LSTM layers skipped.
- SD855 numbers: classifier 2.38 ms; rec 11 ms; det 133 ms after pruning; system 268 ms at 3.5 MB.

**v2–v3** [S2][S3][S14]
- Effort shifted to Intel-CPU / MKLDNN-friendly design: PP-LCNet, and SVTR blocks moved after pooling because MKLDNN handles the Mix block poorly.
- The Paddle-Lite Android demo ships v3 as 16.2 MB FP32 or 5.9 MB slim INT8 .nb files.

**v4** [S4]
- Latency reported on CPU with OpenVINO.

**v5 / 3.x** [S5][S7][S9]
- HPI auto-backend (OpenVINO, ORT, TRT; Paddle→ONNX); Paddle-Lite on-device; PaddleOCR.js in the browser; HF Transformers backend.
- Published latency is T4 and Xeon; no mobile SoC numbers found in 3.x docs `[UNVERIFIED]`.

**v6** [S8]
- First lineage doc to give **Apple M4** numbers: tiny at 0.35 s/img with ORT.
- OpenVINO CPU speedups are large: medium at 1.40 s vs 7.30 s for v5 server.

**VL** [S10][S9]
- GPU serving via vLLM, SGLang and FastDeploy; llama.cpp and MLX-VLM backends added later.
- On-device is not the target.

---

## 6. Trend line and where Baidu is heading

1. **Bag-of-tricks → architecture refresh → data and teachers.**
   - v1–v2 were almost purely tricks: pruning, quantization, distillation, augmentation.
   - v3–v4 renewed the architecture (SVTR, PP-LCNetV3, PFHead) and kept CML/DML/GTC.
   - v5's large jump came mostly from **data scale and VLM-assisted labelling and teaching**, at the cost of a bigger dictionary and slower mobile pipeline [S5][S6].
   - v6 returns to efficiency with a purpose-built MetaFormer-style backbone and tiering [S8].
2. **Recipes that survived every generation:**
   - DB detection with BCE/Dice/L1, until v6 swapped BCE for focal;
   - CTC-only inference with a training-only attention guide (SAR → NRTR);
   - TextConAug;
   - multi-scale rec input with height 48 at eval.

   These are the stable, transferable ideas for a PyTorch port.
3. **Scope widened from Chinese/English** (6,622 chars) → five scripts in one model (v5, about 18.4K entries) → 50 languages in one model (v6, about 18.7K entries) [S1][S15][S8].
4. **Bifurcation.**
   - Baidu now has two product lines:
     - a classic det+rec pipeline (PP-OCRv6, 1.5M–34.5M params) for edge, mobile, CPU and high-throughput line OCR;
     - a **0.9B VLM** (PaddleOCR-VL 1.0 → 1.5 → 1.6) for page-level document parsing (tables, formulas, charts, reading order), served on GPU.
   - The VL line still depends on a classic detector for layout (PP-DocLayoutV2/V3, RT-DETR) and was bootstrapped with labels from the classic pipeline (PP-StructureV3) [S10].
   - Baidu explicitly benchmarks PP-OCRv5/v6 against VLMs and claims lightweight specialists still win at line-level OCR [S5][S8].
5. **What stays on-device:** the PP-OCR tiers.
   - v6 tiny is about 6 MB det+rec on disk [S7]; v5 is available in-browser via PaddleOCR.js [S9]; Paddle-Lite remains the documented on-device path [S5].
   - Nothing in the sources positions PaddleOCR-VL for phones. llama.cpp and MLX support [S9] hint at desktop/Apple-silicon local use `[UNVERIFIED as a strategic direction]`.

---

## 7. Open questions / low-confidence items

- PP-OCRv4 release date, and whether its released student det config really omits PFHead and CML (configs show DBHead) `[UNVERIFIED]`.
- PP-OCRv6:
  - no paper;
  - per-tier parameter split between det and rec (0.43M tiny det vs 1.5M tiny total);
  - "2.37× GPU speedup" claim;
  - whether VLM pseudo-labels or distillation were used for v6 det;
  - medium det neck naming (doc "RepLKFPN" vs config `RepLKPAN`).

  All `[UNVERIFIED]`.
- PP-OCRv5 "0.07B parameters" is quoted from the report's VLM comparison, apparently for the server det+rec system. The exact count per model is `[UNVERIFIED]`.
- Dictionary sizes for v5 and v6 are **line counts of the dict files**. The paper-reported v1 count is 6,622, while `ppocr_keys_v1.txt` has 6,623 lines [S1][S15].
- PaddleOCR-VL-1.5 use of PP-DocLayoutV3 `[UNVERIFIED]`; PaddleOCR-VL-1.6 has no paper.
- PP-OCRv2 recognition input size (assumed unchanged from v1) `[UNVERIFIED]`.

---

## Sources

- **[S1]** Du et al., *PP-OCR: A Practical Ultra Lightweight OCR System*, arXiv:2009.09941 — https://arxiv.org/abs/2009.09941
- **[S2]** Du et al., *PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System*, arXiv:2109.03144 — https://arxiv.org/abs/2109.03144
- **[S3]** Li et al., *PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System*, arXiv:2206.03001 — https://arxiv.org/abs/2206.03001
- **[S4]** PaddleOCR docs, *PP-OCRv4 introduction* — https://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html
- **[S5]** Cui et al., *PaddleOCR 3.0 Technical Report*, arXiv:2507.05595 — https://arxiv.org/abs/2507.05595
- **[S6]** PaddleOCR docs, *PP-OCRv5 introduction* — https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html
- **[S7]** PaddleOCR docs, *General OCR pipeline* (model list tables and test conditions) — https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md
- **[S8]** PaddleOCR docs, *PP-OCRv6 introduction* — https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html (mirror: https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)
- **[S9]** PaddleOCR GitHub releases (v3.2.0 to v3.7.0) — https://github.com/PaddlePaddle/PaddleOCR/releases
- **[S10]** Cui et al., *PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model*, arXiv:2510.14528 — https://arxiv.org/abs/2510.14528
- **[S11]** *PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing*, arXiv:2601.21957 — https://arxiv.org/abs/2601.21957
- **[S12]** Official configs, `PaddlePaddle/PaddleOCR@main`:
  - https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/PP-OCRv4
  - https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/PP-OCRv4
  - https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/PP-OCRv5
  - https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/PP-OCRv5
  - https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/PP-OCRv6
  - https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/PP-OCRv6
- **[S13]** PaddleOCR docs, *PP-OCRv3 introduction* — https://www.paddleocr.ai/main/en/version2.x/ppocr/blog/PP-OCRv3_introduction.html
- **[S14]** PaddleOCR docs (legacy), *Paddle-Lite on-device deployment* — https://www.paddleocr.ai/main/en/version2.x/legacy/lite.html
- **[S15]** Dictionary files (line counts taken 2026-11-10):
  - https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/ppocr_keys_v1.txt
  - https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv5_dict.txt
  - https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv6_dict.txt
  - https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv6_tiny_dict.txt
