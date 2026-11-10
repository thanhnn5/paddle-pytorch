# 04 — Loss functions and training objectives for lightweight OCR

> **Supervisor note (2026-11-10):** some claims in this raw note were corrected after fact-checking. See section 7 of [../README.md](../README.md) and [90-verification-log.md](90-verification-log.md); where they differ, the log is right.

Track: how recognition (mostly CTC) and detection (DB) losses have changed over time, and which objective changes are cheap to try in our PyTorch port of PaddleOCR.
Written 2026-11-10. Every number links to the source it came from. Anything I could not confirm from a primary source is tagged `[UNVERIFIED]`.

**Evidence levels**
- **strong**: ablation in a peer-reviewed paper.
- **moderate**: toolkit ablation, such as PaddleOCR docs or tech reports, often on private data.
- **weak**: a single report, a secondary summary, or a result from another domain such as ASR.

**What the codebase already has.** `torchocr/losses/` contains `ace_loss`, `center_loss`, `rec_enhanced_ctc_loss`, `rec_multi_loss` (CTC + SAR/NRTR), `distillation_loss` and `kd_loss`. `distillation_loss` covers DML, KL, DKD, NRTR-DML, DilaDB, `CTCDKDLoss`, `KLCTCLogits` and `DistillCTCLogits`. `KLCTCLogits` is the PP-OCRv4 approach of averaging CTC logits over time with blank masking. The folder also has `stroke_focus_loss`, `text_focus_loss`, `det_db_loss` (alpha=5, beta=10, ohem_ratio=3) and `aux_distill`, which is per-stage cosine feature distillation with 1×1 adapters. Items marked **NEW** below are not in this list.

---

## 1. CTC and its variants

### 1.1 Vanilla CTC
- **Formula:** $\mathcal L_{CTC} = -\log \sum_{\pi \in \mathcal B^{-1}(y)} \prod_t p(\pi_t\mid x)$
- **Known problems:**
  - It optimizes the whole sequence and gives no per-character or per-frame supervision ([DCTC](https://arxiv.org/html/2308.08806v4)).
  - Its outputs are **peaky**, dominated by blank. Zeyer et al. show that peaky behaviour follows from the CTC criterion itself when gradient descent starts from a uniform initialization, and that adding a label prior removes it ([Zeyer 2021](https://arxiv.org/abs/2105.14849)).
  - Peakiness is harmless at greedy-decode time but breaks frame-level KD (see 4.6).

### 1.2 Focal-CTC
- **Formula:** $\mathcal L = \alpha(1-y')^\gamma \mathcal L_{CTC}$, where $y' = e^{-\mathcal L_{CTC}}$. Settings are α=1, γ=2 ([PaddleOCR Enhanced-CTC doc](https://www.paddleocr.ai/main/version2.x/ppocr/blog/enhanced_ctc_loss.html)).
- **Problem addressed:** sample-level hard mining. The original paper's motivation was rare characters in Chinese.
- **Gain:**
  - The original paper reports +3 to +9 points over CTC on synthetic and real unbalanced Chinese sets ([Feng et al. 2019, *Complexity*](https://doaj.org/article/8fc834f8bc5c4bb5ab5c93873dee3364)).
  - PaddleOCR reports +0.3% on its internal data ([doc](https://www.paddleocr.ai/main/version2.x/ppocr/blog/enhanced_ctc_loss.html)).
  - The independent re-implementation in DCTC got −1.7 (CRNN, English), ±0 (CRNN, Chinese) and −0.2/−0.5 (SVTR-T) ([DCTC Table](https://arxiv.org/html/2308.08806v4)).
- **Cost:** 0 at inference.
- **Evidence:** weak to moderate, and the results conflict. **Already in the codebase.**

### 1.3 A-CTC (CTC + ACE)
- **Formula:** $\mathcal L_{CTC} + \lambda\,\mathcal L_{ACE}$ with λ=0.1 ([doc](https://www.paddleocr.ai/main/version2.x/ppocr/blog/enhanced_ctc_loss.html)).
- **ACE term:** $\mathcal L_{ACE} = -\sum_c \bar N_c \log \bar y_c$, where $\bar y_c = \frac1T\sum_t y_{t,c}$ and $\bar N_c$ is the normalized character count ([Xie et al. CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Aggregation_Cross-Entropy_for_Sequence_Recognition_CVPR_2019_paper.html)).
- **Gain:** +0.7% on PaddleOCR internal data.
- **Cost:** 0.
- **Evidence:** moderate. **Already in the codebase.**

### 1.4 C-CTC (CTC + Center loss)
- **Formula:** $\mathcal L_{CTC} + \lambda\sum_t \lVert x_t - c_{\hat y_t}\rVert^2$, where $\hat y_t$ comes from greedy argmax.
  - The PP-OCRv2 paper uses λ=0.05 ([PP-OCRv2](https://ar5iv.labs.arxiv.org/html/2109.03144)).
  - The later doc uses λ=0.25.
  - Centers are initialized from correctly recognized samples, because random initialization gave no clear gain ([doc](https://www.paddleocr.ai/main/version2.x/ppocr/blog/enhanced_ctc_loss.html)).
- **Problem addressed:** look-alike glyphs. The gain is larger with the 6,625-character Chinese set.
- **Gain:**
  - +1.7% on internal data ([doc](https://www.paddleocr.ai/main/version2.x/ppocr/blog/enhanced_ctc_loss.html)).
  - In PP-OCRv2 recognition, accuracy went 73.9→74.8 (+0.9), on top of U-DML ([PP-OCRv2](https://ar5iv.labs.arxiv.org/html/2109.03144)).
- **Cost:** 0.
- **Evidence:** moderate. **Already in the codebase.**

### 1.5 EnCTC (maximum-entropy regularized CTC)
- **Formula:** $\mathcal L_{CTC} - \beta\,H(p(\pi\mid x))$, an entropy bonus over alignment paths. The paper also adds an equal-spacing pruning variant ([Liu et al. NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/e44fea3bec53bcea3b7513ccef5857ac-Abstract.html); code: [enctc.crnn](https://github.com/liuhu-bigeye/enctc.crnn)).
- **Problem addressed:** peaky, over-confident CTC outputs.
- **Gain:** the original paper reports gains on scene text. DCTC's re-implementation with β=0.2 found −0.2 (CRNN, English), **−4.5 (CRNN, Chinese)** and −0.2/−1.1 (SVTR-T) ([DCTC](https://arxiv.org/html/2308.08806v4)).
- **Cost:** 0.
- **Evidence:** strong that it can work on CRNN in English. It does not reproduce on modern models or on Chinese. **NEW, but not recommended.**

### 1.6 CTC with label smoothing or label priors
- **Label smoothing on CTC frames** (a uniform-KL term on per-frame posteriors) appears in ASR toolkits. I found no OCR ablation for it. `[UNVERIFIED]` for OCR.
- **Label priors:** subtracting a log label prior during training gives less peaky outputs. Results come from ASR forced alignment only, with 12–40% better boundary error ([Huang et al. ICASSP 2024](https://arxiv.org/html/2406.02560v2)). Accuracy gains for OCR are unknown.
- **Evidence:** weak for OCR.

### 1.7 DCTC: Self-distillation Regularized CTC (AAAI 2024)
- **Formula:** $\mathcal L = \mathcal L_{CTC} + \lambda\,\big(-\sum_t \log P(z^*_t, t)\big)$
  - $z^*$ is the frame-wise MAP alignment, computed in closed form from the CTC gradient: $z^*_t = \arg\min_k G_{t,k}/P_{t,k}$, with $G = \partial\mathcal L_{CTC}/\partial U$.
  - The model's own CTC gradient therefore supplies per-frame pseudo-labels. No teacher is needed.
  - λ = 0.025 for English and 0.01 for Chinese ([arXiv 2308.08806](https://arxiv.org/html/2308.08806v4); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28575); code: github.com/zzyhlyoko/DCTC).
- **Problem addressed:** CTC lacks per-character learning, and naive frame-level self-KD or teacher KD is inconsistent. Using the model's own argmax as frame targets *hurt* badly: CRNN went 77.3→75.0 in English and 68.7→61.6 in Chinese ([DCTC](https://arxiv.org/html/2308.08806v4)).
- **Gain** (average accuracy, baseline → DCTC):

  | Model | English | Chinese |
  |---|---|---|
  | CRNN | 77.3→79.9 (+2.6) | 68.7→70.8 (+2.1) |
  | SVTR-T | 90.8→91.7 (+0.9) | 75.3→75.9 (+0.6) |
  | SVTR-S | +0.9 | +1.0 |

  Training followed PaddleOCR configs ([DCTC](https://arxiv.org/html/2308.08806v4)).
- **Cost:**
  - 0 at inference.
  - Training needs the CTC gradient with respect to logits on a detached copy, which costs one extra backward through the CTC op only. The authors wrote a CUDA extension. In PyTorch this could be done with `torch.autograd.grad` on a detached logits copy. `[UNVERIFIED: PyTorch port effort]`
- **Evidence:** strong. **NEW. This is the most directly relevant new item for a small CTC recognizer.**

### 1.8 Intermediate / auxiliary CTC
- **Formula:** $(1-w)\mathcal L_{CTC}(\text{final}) + w\,\mathcal L_{CTC}(\text{intermediate layer})$, using a shared or separate classifier ([Lee & Watanabe ICASSP 2021](https://arxiv.org/abs/2102.03216)).
- **Results:** ASR only, reported as giving gains with no inference overhead. I found no OCR ablation.
- **Cost:** 0 if the auxiliary head is dropped at inference.
- **Evidence:** strong in ASR, none in OCR. **NEW.** It could attach to the neck or penultimate SVTR block.

### 1.9 CR-CTC: consistency-regularized CTC (ICLR 2025)
- **Formula:** $\mathcal L = \tfrac12[\mathcal L_{CTC}(z^a,y)+\mathcal L_{CTC}(z^b,y)] + \alpha\cdot\tfrac12\sum_t[\mathrm{KL}(\mathrm{sg}(z^b_t)\Vert z^a_t)+\mathrm{KL}(\mathrm{sg}(z^a_t)\Vert z^b_t)]$
  - $a$ and $b$ are two independently augmented views of the same input. α=0.2 by default ([Yao et al.](https://arxiv.org/html/2410.05101v4)).
- **Problem addressed:** the paper gives three explanations:
  - self-distillation between random sub-models;
  - masked prediction;
  - suppression of peaky distributions. Non-blank tokens average 1.28 frames vs 1.04 for CTC, and blank probability drops from 99.64% to 94.19% ([paper](https://arxiv.org/html/2410.05101v4)).
- **Gain (ASR):** Zipformer-M on LibriSpeech went from 2.51/6.02 to 2.12/4.62 WER ([paper](https://arxiv.org/html/2410.05101v4)).
- **Cost:**
  - 0 at inference.
  - Training needs 2 forward passes. The authors halve the batch and epochs to keep compute comparable.
- **OCR use:** I found no OCR, STR or HTR application. Searched 2026-11-10; for OCR it is untested, which is an opportunity rather than evidence.
- **Spatial augmentation caveat:** frame-wise KL requires the two views to share a time axis. For OCR, use photometric or blur augmentation, or masking, not geometric augmentation that shifts columns. This is my inference; tag `[UNVERIFIED]`.
- **Evidence:** strong in ASR, none in OCR. **NEW.**

### 1.10 Blank-handling tricks
- **PP-OCRv4 `KLCTCLogits`:** masks the blank column and averages softmax over time before the KL ([PP-OCRv4 doc](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)). Already in the codebase.
- **ASR blank-selection study:** naive blank elimination "does not always work off the shelf". *Symmetric* blank-frame selection lets the student drop the CTC loss during KD with little or no degradation, which allows label-free distillation ([Hilmes et al. Interspeech 2025](https://arxiv.org/abs/2506.01503)). No numbers were checked, only the abstract.
- **Evidence:** weak for OCR.

---

## 2. Attention / CE-based objectives

- **Label-smoothed CE (NRTR/SAR heads):** PP-OCRv4 found that **removing** label smoothing from the NRTR-head CE term improved accuracy when that term was combined with DKD distillation ([PP-OCRv4 doc](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)). The size of the effect is not reported. Evidence: moderate. Check whether our `rec_multi_loss` and NRTR configs have smoothing on.
- **Permutation LM loss (PARSeq):**
  - Formula: $\mathcal L = \frac1K\sum_{k=1}^K \mathcal L_{ce}(y_k, \hat y)$ over K attention-mask permutations. The left-to-right order is always included, and permutations come in flipped pairs. K=6 ([PARSeq ECCV 2022](https://arxiv.org/abs/2207.06966), Eq. 6).
  - Ablation (Table 3), K=1→6: AR accuracy 93.04→93.34 and NAR accuracy 0.01→92.22. Training time rises from 5.86 to 8.48 hours.
  - Relevant to us only as a teacher objective or GTC branch. Evidence: strong.
- **CPPD counting and ordering losses:**
  - Formula: $\mathcal L = \lambda_{co}\mathcal L_{co} + \lambda_{cc}\mathcal L_{cc} + \lambda_{rec}\mathcal L_{rec}$, with all λ set to 1 ([CPPD arXiv 2307.12270](https://arxiv.org/abs/2307.12270v2)).
    - CC is a per-class count classification: $-\frac1C\sum_c\sum_l y_{c,l}\log p_{c,l}$.
    - CO is a binary CE on whether each position is occupied.
  - Gain (SVTR-B-CPPD, English):
    - Both modules and both losses reach 93.80 against a 92.01 baseline.
    - The CC loss beat ACE on the same slot (93.80 vs 93.31).
    - CO-BCE beat plain CE (93.80 vs 93.50) (Tables 2–3).
  - **Takeaway for CTC:** the CC loss is a stronger "count" auxiliary than ACE. It needs a small training-only query head. Evidence: strong, but measured on parallel decoders, not CTC.
- **ACE:** see 1.3.
- **2D attention supervision:** I found no recent primary source with an ablation useful for edge CTC. `[UNVERIFIED]`, skipped.

---

## 3. Hybrid / multi-head (training-only guide branches)

| Method | Objective | Gain | Inference cost | Evidence |
|---|---|---|---|---|
| GTC (AAAI 2020) | CTC + attention-decoder loss on a shared encoder; the attention branch guides alignment ([arXiv 2002.01276](https://arxiv.org/abs/2002.01276)) | Paper reports SOTA at the time and ~6× faster than attention models | 0 (branch dropped) | strong |
| PP-OCRv3 GTC (CTC + SAR) | MultiHead loss | 74.0→75.8 (+1.8) ([PP-OCRv3 T4](https://ar5iv.labs.arxiv.org/html/2206.03001)) | 0 | moderate |
| PP-OCRv4 GTC-NRTR | SAR guide replaced by NRTR; reported as more stable | Lite-Neck + GTC-NRTR: 72.70→73.21 ([v4 doc](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)) | 0 | moderate |
| PP-OCRv5 | Same GTC-NRTR guiding SVTR-HGNet; no component ablation published ([PaddleOCR 3.0 report](https://arxiv.org/html/2507.05595v1)) | n/a | 0 | weak |
| **SVTRv2 SGM** | $\mathcal L = 0.1\,\mathcal L_{ctc} + 1.0\,\mathcal L_{sgm}$. The SGM term is the average CE of predicting $c_i$ from its left and right context strings, each embedded as a query that attends over visual features ([SVTRv2](https://arxiv.org/html/2411.15858v2)) | Without SGM → SGM: OST 74.92→80.03, U14M 83.86→86.14. GTC in the same setup gave only 75.70 / 84.33 | 0 ("discarded for efficiency") | strong (ICCV 2025) |

- **Takeaway:** SGM beats GTC by about 4.3 OST and about 1.8 U14M points in a controlled ablation (my arithmetic from [Table 2](https://arxiv.org/html/2411.15858v2)). **SGM is NEW relative to our `rec_multi_loss`, which is CTC + SAR/NRTR.**
- **Caveat:** these gains were measured on SVTRv2 encoders. Transfer to a tiny conv backbone is untested. `[UNVERIFIED]`
- **Joint CTC-attention weighting:** PaddleOCR MultiHead uses equal weights by default `[UNVERIFIED: check yml]`. SVTRv2 notably *down-weights* CTC to 0.1.

---

## 4. Knowledge-distillation objectives

### 4.1 DML / U-DML (PP-OCRv2 and v3)
- **Formula:**
  - $\mathcal L_{ctc} = CTC(S)+CTC(T)$
  - $\mathcal L_{dml} = \tfrac12[KL(S\Vert T)+KL(T\Vert S)]$ on per-frame softmax
  - $\mathcal L_{feat} = L2(S_{backbone}, T_{backbone})$
  - All weights are 1 ([PP-OCRv2](https://ar5iv.labs.arxiv.org/html/2109.03144)).
- **Gain:**
  - PP-OCRv2: 69.3→73.9 (+4.6).
  - PP-OCRv3, with an attention head added to the mutual learning: 76.9→78.4 (+1.5) ([PP-OCRv3](https://ar5iv.labs.arxiv.org/html/2206.03001)).
- **Cost:** 0.
- **Evidence:** moderate. **Already in the codebase.**

### 4.2 CML (detection, PP-OCRv2/v3/v4)
- **Formula:** $\mathcal L = \mathcal L_{gt} + \mathcal L_{dml} + \mathcal L_{distill}$
  - $\mathcal L_{dml}$ is the symmetric KL between two student probability maps.
  - $\mathcal L_{distill} = \gamma\, l_p(S, f_{dila}(T)) + l_b(S, f_{dila}(T))$, with γ=5. $f_{dila}$ is a 2×2 dilation of the teacher's binarized map ([PP-OCRv2](https://ar5iv.labs.arxiv.org/html/2109.03144)).
- **Gain:**
  - PP-OCRv2: Hmean 0.759→0.789 (+3.0). DML alone gave +1.8.
  - PP-OCRv3: DB-MV3 81.3 (no CML) → 84.3 with CML. The teacher is DB-R50-LK-PAN-DML at 86.0 ([PP-OCRv3 T2](https://ar5iv.labs.arxiv.org/html/2206.03001)).
  - PP-OCRv4 added a KL term between student and teacher response maps. The text says 79.08→79.56, but the table says 79.87; the doc contradicts itself ([v4 doc](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)).
- **Evidence:** moderate. **Already in the codebase** (DistillationDilaDBLoss, DML).

### 4.3 DKD (decoupled KD, CVPR 2022)
- Splits KL into target-class (TCKD) and non-target-class (NCKD) parts, each with its own weight ([DKD arXiv 2203.08679](https://arxiv.org/abs/2203.08679)). I did not re-fetch the formula weights.
- PP-OCRv4 applies DKD on the NRTR head, combined with blank-masked time-averaged CTC-logit KL: 74.72→75.45 (+0.73) ([v4 doc](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)).
- **Evidence:** moderate. **Already in the codebase** (`CTCDKDLoss`, `DistillationDKDLoss`).

### 4.4 Feature-map L2 and cosine
- U-DML's backbone L2 term, plus our `aux_distill` (cosine with 1×1 adapters, RT-DETRv4-style). Already covered.

### 4.5 Attention-map, hint and affinity distillation for text
- "Text is Text, No Matter What" (ICCV 2021) proposes three distillation losses for variable-length text. Secondary summaries describe them as character-aligned glimpse-vector hints, attention-map distillation and affinity distillation ([arXiv 2107.12087](https://arxiv.org/abs/2107.12087)). The loss names come from a secondary summary: `[UNVERIFIED in abstract]`. No numbers checked.
- **Evidence:** weak for our use. It needs attention decoders on both sides.

### 4.6 CTC-posterior distillation and its pitfalls
- Frame-wise KL between two CTC models assumes they share a frame alignment. This fails across architectures and topologies ([ASR literature summary via Hilmes et al.](https://arxiv.org/abs/2506.01503)).
- Fixes in the literature:
  - (a) time-averaged or blank-masked logits, as in PP-OCRv4 `KLCTCLogits`;
  - (b) sequence-level KD, where teacher n-best or greedy outputs become pseudo-labels;
  - (c) MAP-alignment self-distillation (DCTC);
  - (d) the Oracle Teacher, which conditions the teacher on labels to get cleaner alignments and was tested on STR ([arXiv 2111.03664](https://arxiv.org/abs/2111.03664)). I did not verify its numbers.
- **Evidence for the failure mode in OCR:** DCTC showed that "Teacher" frame targets from a transformer branch *lowered* Chinese CRNN accuracy to 64.1 from 68.7, and gained only +1.7 in English ([DCTC](https://arxiv.org/html/2308.08806v4)).

### 4.7 Distilling from big transformers or VLMs: pseudo-labels vs soft labels
- **Cloze Self-Distillation (CVPR 2025):**
  - Formula: NLL on **teacher pseudo-labels**, which replace the ground truth, plus $\alpha\,KL_\tau$(teacher cloze distribution ‖ student). α=0.1, τ=2, and results are insensitive to both ([arXiv 2503.16184](https://arxiv.org/html/2503.16184v1)).
  - Gain (ViT-B, AVG11, Real): supervised 92.0, pseudo-labels 92.3, pseudo-labels + KD 92.5.
  - Larger gains in low-data settings: at 10% of the data, 89.1 vs 86.9.
  - Motivation: real STR labels are noisy.
  - The teacher and student are the same architecture (PLM decoder), not a CTC student.
  - **Evidence:** strong. Gains are modest at full data.
- **VLM → CTC student:**
  - PP-OCRv5 uses a GOT-OCR2.0 visual encoder as a feature teacher for the **detector** (PP-HGNetV2 student). The loss type and gain are unpublished ([PaddleOCR 3.0 report](https://arxiv.org/html/2507.05595v1)).
  - I found no peer-reviewed ablation of a VLM teacher distilled into a CTC recognizer via soft labels. Pseudo-labelling with VLM outputs is common practice, but the evidence is `[UNVERIFIED]`.
- **CLIP-OCR (ACM MM 2023):** distils CLIP linguistic features into the recognition decoder ([arXiv 2310.04999](https://arxiv.org/abs/2310.04999)). Details not checked.

### 4.8 Cross-architecture tricks
- **Projectors and adapters:** we already have 1×1 adapters in `aux_distill`.
- **Relational / affinity KD:** see 4.5. FGD's global relation term is also relational (5.7).
- No OCR-specific ablation found beyond these. `[UNVERIFIED]`

---

## 5. Auxiliary / perceptual losses

- **Text-focused and stroke-focused SR losses:**
  - Scene Text Telescope (CVPR 2021) uses position-aware and content-aware losses, with weighted CE for confusable characters. Both are training-only ([CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scene_Text_Telescope_Text-Focused_Scene_Image_Super-Resolution_CVPR_2021_paper.html)).
  - Text Gestalt (AAAI 2022) has a stroke-focused module that is training-only, so "no time overhead at test" ([arXiv 2112.08171](https://arxiv.org/abs/2112.08171)).
  - These target SR networks. Using them as auxiliary losses on a recognizer needs an SR branch.
  - **Already in the codebase** (`text_focus_loss`, `stroke_focus_loss`).
- **SR branch used only at training time in PP-OCRv4 server:** I found **no** support for this in the PP-OCRv4 doc. The recognition ablation lists DF, Lite-Neck, GTC-NRTR, PP-LCNetV3, multi-scale, TextConAug and DKD only ([v4 doc](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)). `[UNVERIFIED — likely false]`
- **KD-LTR (low-resolution text recognition via KD from high-resolution images):** [arXiv 2308.02770](https://arxiv.org/abs/2308.02770). Numbers not checked.
- **Contrastive losses:**
  - SeqCLR (CVPR 2021): frame-window contrastive pre-training. Big gains in semi-supervised handwriting ([CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Aberdam_Sequence-to-Sequence_Contrastive_Learning_for_Text_Recognition_CVPR_2021_paper.html)). Exact numbers not checked.
  - ConCLR (AAAI 2022): pulls together the same character across concatenated contexts, which improves out-of-vocabulary recognition ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20245)). Numbers not checked. Designed for attention recognizers.
  - Evidence: strong as papers, but low priority for a supervised CTC edge model.
- **Character-count auxiliary head:** CPPD's CC loss beats ACE (2). PaddleOCR's TextConAug (+0.5) is augmentation, not a loss.

---

## 6. Imbalance and long-tail vocabulary

- **Focal-CTC:** mixed evidence (1.2).
- **Center loss for similar glyphs:** moderate evidence, larger gain with big charsets (1.4).
- **Data filtering by confidence** (PP-OCRv4 DF, a sampling policy rather than a loss):
  - Drop samples with confidence above 0.95 (too easy) and below 0.15 (likely mislabeled).
  - Result: 71.50→72.70 (+1.2), and training time cut from 2 weeks to 5 days ([v4 doc](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)). Evidence: moderate.
- **Pseudo-label correction of noisy labels:** Cloze-SD gives +0.3 with pseudo-labels alone (4.7).
- **Class-balanced re-weighting of CTC by character frequency:** I found no OCR ablation. `[UNVERIFIED]`
- **PP-OCRv5 "balanced sampling":** only in a secondary summary. `[UNVERIFIED]`

---

## 7. Detection-side loss tweaks (brief)

- **DB loss:**
  - Formula: $L = L_s + \alpha L_b + \beta L_t$ with **α=1, β=10** in the paper ([DB, arXiv 1911.08947](https://ar5iv.arxiv.org/html/1911.08947)).
  - $L_s$ and $L_b$ are BCE with OHEM at positive:negative = 1:3. $L_t$ is L1 inside the dilated polygon.
  - PaddleOCR and our `det_db_loss` default to **α=5**, β=10, ohem_ratio=3. PaddleOCR's binary-map loss uses dice by default `[UNVERIFIED: check main_loss_type in yml]`. The α=5 also appears in PP-OCRv2's CML $\mathcal L_{gt}$ ([PP-OCRv2](https://ar5iv.labs.arxiv.org/html/2109.03144)).
  - I found no published ablation of α. Evidence: weak.
- **Dynamic shrink ratio** (a label-generation change, not a loss): ratio ramps from 0.4 to 0.6 over training. 76.97→78.24 (+1.27) in PP-OCRv4 ([v4 doc](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)). Evidence: moderate. **NEW and cheap.**
- **CML:** DML on maps plus a dilated-teacher-supervised term (4.2).
- **Foreground/background-weighted feature KD:**
  - FGD (CVPR 2022) separates foreground and background with GT-box masks and teacher spatial/channel attention, and adds a global relation term.
  - +2.9 to +3.6 mAP on COCO students ([arXiv 2111.11837](https://arxiv.org/abs/2111.11837), [code](https://github.com/yzd-v/FGD)).
  - For text, the foreground mask can be the shrink map. This is untested on text detection `[UNVERIFIED]`.
  - Our `aux_distill` cosine loss is currently uniform over pixels. Adding a mask is a small change.
  - Evidence: strong for generic detection, weak for text.
- **Kernel-mask KD for segmentation text detectors:** journal paper ([Complex & Intelligent Systems 2023](https://doaj.org/article/aa64cdd864b54b80886cb3f260c1aaa3)). Details not verified.

---

## Summary table

| Loss / objective | Target problem | Evidence | Reported gain | Inference cost | In codebase? |
|---|---|---|---|---|---|
| Focal-CTC | hard or rare samples | weak–moderate, conflicting | +0.3 (Paddle), −1.7…0 (DCTC re-impl) | 0 | yes |
| A-CTC (ACE aux) | count supervision | moderate | +0.7 | 0 | yes |
| C-CTC (center) | look-alike glyphs | moderate | +0.9 to +1.7 | 0 | yes |
| EnCTC | peaky CTC | strong (CRNN-En only); fails on Chinese and SVTR | −4.5…+ | 0 | no |
| **DCTC** | per-frame supervision for CTC | **strong** | **+0.6–2.6** | 0 | **no** |
| **CR-CTC** | peaky CTC / overfitting | strong (ASR), none (OCR) | ASR WER −16% to −24% rel. | 0 (2× train forward) | **no** |
| Intermediate CTC | deep supervision | strong (ASR) | n/a for OCR | 0 | **no** |
| GTC (SAR/NRTR guide) | CTC alignment | moderate–strong | +1.8 (v3), +0.5 (v4) | 0 | yes |
| **SVTRv2 SGM** | linguistic context into CTC features | **strong** | **+5.1 OST, +2.3 U14M; beats GTC** | 0 | **no** |
| CPPD CC loss | counting aux | strong (on PD) | CC beats ACE by 0.49 | 0 if head dropped | **no** |
| PARSeq PLM | AR/NAR unification | strong | K=6 makes NAR work | n/a | no (teacher only) |
| U-DML | mutual learning | moderate | +4.6 (v2), +1.5 (v3) | 0 | yes |
| CML (det) | det distillation | moderate | +3.0 Hmean | 0 | yes |
| DKD + time-avg CTC KL | KD across CTC alignments | moderate | +0.73 | 0 | yes |
| Cloze-SD pseudo-label + KD | label noise | strong | +0.5 (full data), +2.2 (10% data) | 0 | **no** |
| No label smoothing in NRTR CE under KD | KD interplay | moderate | unquantified | 0 | check config |
| PP-OCRv4 data filter | noisy or easy data | moderate | +1.2 | 0 | **no** |
| Dynamic shrink ratio (det) | label geometry | moderate | +1.27 Hmean | 0 | **no** |
| FGD-style masked feature KD | uniform feature KD wastes capacity on background | strong (COCO), none (text) | +2.9–3.6 mAP (COCO) | 0 | partial (`aux_distill` unmasked) |

---

## Ranked shortlist: cheap objective changes for a small CTC recognizer and a DB detector distilled from a strong teacher

Expected gains are my extrapolations from the cited ablations to our setting, not measurements.

1. **DCTC self-distillation term on the CTC head.**
   - Expected gain: +0.5 to +1.5 accuracy for SVTR/LCNet-class models; up to +2.6 seen on CRNN.
   - Cost: ~40 lines; one extra CTC-gradient computation per step.
   - Risk: **low–medium**. The pitfall is getting the $G/P$ argmin right with PyTorch's CTC gradient and log-softmax conventions. Verify that pseudo-labels on a converged model match the greedy path for most frames.
   - Composes with existing teacher KD.
2. **Turn off label smoothing in the NRTR/SAR guide CE when distilling, and audit the MultiHead weights.**
   - Expected gain: small, possibly +0.2 to +0.5 `[UNVERIFIED magnitude]`.
   - Risk: **very low**; it is a config flip.
3. **CR-CTC consistency loss (two photometric/masking views, symmetric frame KL with stop-gradient, α≈0.2).**
   - Expected gain: unknown for OCR. ASR shows large relative gains, and the method directly attacks peakiness, which also helps later CTC-to-CTC distillation.
   - Risk: **medium**. It doubles forward cost. Views must keep the time axis aligned, so no horizontal shift or crop.
   - Untested on OCR, so a cheap ablation would be novel.
4. **Data filtering by teacher confidence plus teacher pseudo-label replacement (Cloze-SD style) for noisy real data.**
   - Expected gain: +0.3 to +1.2.
   - Risk: **low**. It needs an offline pass with the teacher. Keep the threshold conservative so rare-but-correct samples are not filtered out.
5. **Foreground-masked `aux_distill` for the DB student.** Weight the cosine or L2 by the dilated shrink map plus a small background weight, and optionally add an FGD-style attention mask.
   - Expected gain: +0.3 to +1.0 Hmean `[UNVERIFIED — extrapolated from COCO]`.
   - Risk: **low**; a small change to the existing module.
6. **Dynamic shrink ratio (0.4→0.6) for the DB label.**
   - Expected gain: about +1 Hmean in PP-OCRv4.
   - Risk: **low**. It is a label-generator change; post-processing unclip must stay consistent with the final ratio.
7. **SVTRv2-style SGM as a training-only branch (replacing or alongside GTC-NRTR; CTC weight 0.1, SGM 1.0).**
   - Expected gain: potentially the largest (+2 to +5 on hard and OOD sets in SVTRv2), but unproven on tiny conv backbones.
   - Risk: **medium–high**: new module, hyperparameters, and the CTC down-weighting interacts with KD.
8. **CPPD-style character-count head** (replacing the ACE term in A-CTC with the CC loss).
   - Expected gain: +0.2 to +0.5 `[UNVERIFIED for CTC]`.
   - Risk: low–medium.

**Deprioritize:**
- EnCTC: it fails on Chinese and SVTR in an independent re-implementation.
- Focal-CTC as a default: conflicting results.
- Naive frame-level KL between CTC models with different architectures: it hurt in DCTC's test.
- Contrastive pre-training (SeqCLR/ConCLR): it matters mainly in the low-label regime.

---

## Sources

- PaddleOCR Enhanced CTC doc: https://www.paddleocr.ai/main/version2.x/ppocr/blog/enhanced_ctc_loss.html
- PP-OCRv2 (arXiv 2109.03144): https://ar5iv.labs.arxiv.org/html/2109.03144
- PP-OCRv3 (arXiv 2206.03001): https://ar5iv.labs.arxiv.org/html/2206.03001
- PP-OCRv4 intro (PaddleOCR docs): https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html
- PaddleOCR 3.0 / PP-OCRv5 report (arXiv 2507.05595): https://arxiv.org/html/2507.05595v1
- DCTC, AAAI 2024 (arXiv 2308.08806): https://arxiv.org/html/2308.08806v4 ; https://ojs.aaai.org/index.php/AAAI/article/view/28575
- CR-CTC, ICLR 2025 (arXiv 2410.05101): https://arxiv.org/html/2410.05101v4
- Zeyer et al., Why does CTC result in peaky behavior?: https://arxiv.org/abs/2105.14849
- Huang et al., CTC with label priors, ICASSP 2024: https://arxiv.org/html/2406.02560v2
- EnCTC, NeurIPS 2018: https://papers.nips.cc/paper/2018/hash/e44fea3bec53bcea3b7513ccef5857ac-Abstract.html ; https://github.com/liuhu-bigeye/enctc.crnn
- Focal CTC (Feng et al., Complexity 2019): https://doaj.org/article/8fc834f8bc5c4bb5ab5c93873dee3364
- ACE, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Aggregation_Cross-Entropy_for_Sequence_Recognition_CVPR_2019_paper.html
- Intermediate CTC, ICASSP 2021: https://arxiv.org/abs/2102.03216
- Blank in CTC KD, Interspeech 2025: https://arxiv.org/abs/2506.01503
- GTC, AAAI 2020: https://arxiv.org/abs/2002.01276
- SVTRv2, ICCV 2025: https://arxiv.org/html/2411.15858v2
- PARSeq, ECCV 2022: https://arxiv.org/abs/2207.06966
- CPPD: https://arxiv.org/abs/2307.12270v2
- Cloze Self-Distillation, CVPR 2025: https://arxiv.org/html/2503.16184v1
- Oracle Teacher: https://arxiv.org/abs/2111.03664
- Text is Text, No Matter What, ICCV 2021: https://arxiv.org/abs/2107.12087
- CLIP-OCR / symmetrical linguistic feature distillation: https://arxiv.org/abs/2310.04999
- KD-LTR: https://arxiv.org/abs/2308.02770
- DKD, CVPR 2022: https://arxiv.org/abs/2203.08679
- Scene Text Telescope, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scene_Text_Telescope_Text-Focused_Scene_Image_Super-Resolution_CVPR_2021_paper.html
- Text Gestalt, AAAI 2022: https://arxiv.org/abs/2112.08171
- SeqCLR, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Aberdam_Sequence-to-Sequence_Contrastive_Learning_for_Text_Recognition_CVPR_2021_paper.html
- ConCLR, AAAI 2022: https://ojs.aaai.org/index.php/AAAI/article/view/20245
- DBNet (arXiv 1911.08947): https://ar5iv.arxiv.org/html/1911.08947
- FGD, CVPR 2022: https://arxiv.org/abs/2111.11837 ; https://github.com/yzd-v/FGD
- Kernel-mask KD for text detection (2023): https://doaj.org/article/aa64cdd864b54b80886cb3f260c1aaa3
