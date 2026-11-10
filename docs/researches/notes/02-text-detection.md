# 02 — Scene/Document Text Detection: Architecture & Loss Evolution (2019 → late 2026)

> **Supervisor note (2026-11-10):** some claims in this raw note were corrected after fact-checking. See section 7 of [../README.md](../README.md) and [90-verification-log.md](90-verification-log.md); where they differ, the log is right.

_Track owner: research assistant. Compiled 2026-11-10. Scope: real-time / mobile-class detectors, losses, distillation, edge backbones._

**Conventions**
- Numbers carry an inline source link. Unless a link says otherwise, it points to the paper's own tables (arXiv PDF) or the official repo/docs.
- `[UNVERIFIED]` means I could not confirm the claim from a primary source in this session. Treat it as a lead, not a fact.
- FPS figures from different papers are **not comparable**. GPUs differ (1080Ti, 3090, T4+TensorRT, V100), and so do input sizes and whether post-processing is counted. None of the classic text-detection papers report Apple ANE, mobile-GPU or NPU latency.

---

## 1. Timeline table

| Method | Year | Venue | Key idea | Loss | Backbone (as reported) | Reported F / speed |
|---|---|---|---|---|---|---|
| EAST | 2017 | CVPR | Single FCN that regresses a rotated box or quad per pixel; NMS | (score + geometry; details not re-checked) `[UNVERIFIED]` | PVANet/VGG `[UNVERIFIED]` | IC15 F=78.20 @ 13.2 FPS (720p) [abs](https://arxiv.org/abs/1704.03155) |
| PSENet | 2019 | CVPR | Multi-scale kernels; progressive scale expansion separates adjacent text | L = λ·Lc + (1−λ)·Ls, dice, OHEM, λ=0.7 [pdf](https://arxiv.org/pdf/1903.12473) | ResNet-50 + FPN | CTW1500 F=74.3 @ 27 FPS (fast setting); best F=82.2 [abs](https://arxiv.org/abs/1903.12473) |
| PAN | 2019 | ICCV | Light FPEM (cascaded U-shape) + FFM neck; learnable Pixel Aggregation via similarity vectors | L = Ltex + 0.5·Lker + 0.25·(Lagg + Ldis); dice; OHEM 3:1 on Ltex [pdf](https://arxiv.org/pdf/1908.05900) | ResNet-18 | CTW1500 F=79.9 @ 84.2 FPS [abs](https://arxiv.org/abs/1908.05900) |
| DBNet | 2020 | AAAI | Differentiable binarization B̂ = 1/(1+e^{−k(P−T)}), learned threshold map; threshold branch dropped at inference | L = Ls + α·Lb + β·Lt, α=1, β=10; BCE with OHEM 1:3 on Ls and Lb; L1 on Lt inside dilated polygon; shrink r=0.4; unclip r′=1.5 [pdf](https://arxiv.org/pdf/1911.08947) | ResNet-18/50 (+DCN) + FPN | MSRA-TD500 R18 F=82.8 @ 62 FPS; Total-Text R18 F=82.8 @ 50 FPS; IC15 R50(1152) F=87.3 @ 12 FPS (1080Ti) [pdf](https://arxiv.org/pdf/1911.08947) |
| PAN++ | 2021 | TPAMI | Kernel + PA detection with a Masked-RoI attention recognition head (spotting) | — | ResNet-18 + stacked FPEM | Total-Text end-to-end F=64.9 @ 29.2 FPS [abs](https://arxiv.org/abs/2105.00405) |
| FCENet | 2021 | CVPR | Fourier contour embedding; inverse Fourier transform + NMS | L = Lcls + λ·Lreg [pdf](https://arxiv.org/pdf/2104.10442) | ResNet-50 + DCN + FPN | CTW1500 F=85.5, Total-Text F=85.8, IC15 F=86.2 [pdf Tab.1](https://arxiv.org/pdf/2104.10442) |
| CentripetalText (CT) | 2021 | NeurIPS | Text kernels + per-pixel centripetal shifts; Relaxed L1 loss | seg loss + Relaxed L1 (masked Smooth-L1) [pdf](https://arxiv.org/pdf/2107.05945) | ResNet-18 | Total-Text CT-640 F=86.3 @ 40.0 FPS (pretrained); CT-320 F=81.2 @ 93.2 FPS [pdf](https://arxiv.org/pdf/2107.05945) |
| PP-OCRv2 det (CML) | 2021 | arXiv | Collaborative Mutual Learning: frozen teacher + 2 students (DML) | GT DB loss + KL peer loss + teacher distill (BCE + Dice to *dilated* teacher map, γ=5) [pdf](https://arxiv.org/pdf/2109.03144) | Teacher R18; student MobileNetV3-L ×0.5 | +7% end-to-end precision vs PP-OCR [abs](https://arxiv.org/abs/2109.03144) |
| DBNet++ | 2022 | TPAMI | DB + Adaptive Scale Fusion (spatial attention over 4 concatenated scales) | Same as DBNet [pdf](https://arxiv.org/pdf/2202.10304) | ResNet-18/50 | Total-Text R18(800) F=83.3 @ 48 FPS; CTW1500 R18(1024) F=83.9 @ 40 FPS; MSRA-TD500 R50(736) F=87.2 @ 29 FPS (1080Ti) [pdf](https://arxiv.org/pdf/2202.10304) |
| PP-OCRv3 det | 2022 | arXiv | Teacher LK-PAN (PAN kernel 3×3→9×9) + DML; student RSE-FPN; CML | DB loss + CML [docs](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv3_introduction.html) | MobileNetV3 student (3.6M) | Student hmean 83.2→85.4 (internal set) [docs](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv3_introduction.html) |
| TextBPN++ | 2022/23 | T-MM | Boundary transformer that iteratively deforms boundary points | — `[UNVERIFIED]` | ResNet-50 | Repo-author re-impl: Total-Text F≈87.9 @ 13 FPS `[UNVERIFIED — not paper numbers]` ([arXiv](https://arxiv.org/abs/2205.05320)) |
| FAST | 2021 (v2 2023) | arXiv (TPAMI version `[UNVERIFIED]`) | 1-channel Minimalist Kernel Representation; GPU text dilation (max-pool); NAS TextNet with reparameterizable convs | L = Lker + 0.5·Ltex, dice; OHEM on Ltex [pdf](https://arxiv.org/pdf/2111.02394) | TextNet-T/S/B (6.8/8.0/8.9M) | Total-Text FAST-T-448 F=81.6 @ 152.8 FPS; FAST-B-800 F=87.5 @ 46 FPS; >600 FPS with TensorRT (1080Ti) [pdf](https://arxiv.org/pdf/2111.02394) |
| DPText-DETR | 2023 | AAAI | Explicit point queries + EFSA; positional label form | DETR-style (Hungarian) `[UNVERIFIED details]` | ResNet-50 | Total-Text F=89.0, CTW1500 F=88.8 (no FPS) [repo](https://github.com/ymy-k/DPText-DETR) |
| DeepSolo | 2023 | CVPR | DETR-like spotting; explicit ordered points per character | — | ResNet-50 / ViT | No numbers in abstract [abs](https://arxiv.org/abs/2211.10772) |
| PP-OCRv4 det | 2023 | docs | PFHead + Dynamic Shrink Ratio (0.4→0.6) + PP-LCNetV3 + CML with an extra KL on response maps | DB loss + CML + KL | PP-LCNetV3 (4.8M model) | Student hmean 76.22→79.87; 67 ms CPU+MKLDNN [docs](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html) |
| SRFormer | 2024 | AAAI | DETR with segmentation in early decoder layers + Mask-informed Query Enhancement | seg + regression `[UNVERIFIED details]` | ResNet-50 | Total-Text F=90.0, CTW1500 F=89.6 (no FPS) [repo](https://github.com/retsuh-bqw/SRFormer-Text-Det) |
| LRANet | 2024 | AAAI | Low-rank approximation of text contour shape | — | ResNet-50 `[UNVERIFIED]` | CTW1500 F=87.4, Total-Text F=89.0 @ 37.2 FPS (RTX3090), per secondary summary [arXiv](https://arxiv.org/abs/2306.15142) `[UNVERIFIED — not read in paper]` |
| EdgeText | 2025 | arXiv | Long edges fitted as continuous curves; PI-loss for curve params | PI-loss | — | No numbers in abstract [abs](https://arxiv.org/abs/2504.04001) |
| PP-OCRv5 det | 2025/26 | arXiv | DB + PP-LCNetV3 (mobile) / PP-HGNetV2 (server); LK-PAN + RSE-FPN | DB | as listed | PP-OCRv5_mobile avg Hmean 75.2, server 81.6 (measured on PP-OCRv6's in-house set) [PP-OCRv6](https://arxiv.org/html/2606.13108); [PaddleOCR 3.0](https://arxiv.org/abs/2507.05595), [PP-OCRv5](https://arxiv.org/html/2603.24373v1) |
| **PP-OCRv6 det** | 2026 | arXiv | Unified LCNetV4 (reparam DW token mixer); **RepLKFPN** neck (DilatedReparamBlock 7×7); **training-only aux DB heads** at P2–P4 | Shrink map: **Dice + Focal** (α=0.25, γ=2.5); aux heads Dice+Focal + L1 threshold; no det distillation | LCNetV4 0.43M / 2.48M / 15.5M | Avg Hmean Tiny 80.6 / Small 84.1 / Medium 86.2 [html](https://arxiv.org/html/2606.13108) |
| TextDS | 2026 | ECCV | VFM dual encoder + step-wise LoRA, for distribution shift | — | VFM (unnamed in abstract) | 4.9M trainable params; no numbers in abstract [abs](https://arxiv.org/abs/2606.28077) |

---

## 2. Segmentation-based lineage (Item 1)

**PSENet → PAN → DB → DB++ → FAST/CT → PP-OCR v2–v6** is the line that dominates real-time and mobile detection.

- **PSENet** ([CVPR'19](https://arxiv.org/abs/1903.12473)) predicts n kernels and grows them by BFS from the smallest. It separates touching text well, but the post-processing is serial and slow on CPU: 27 FPS in its fastest setting with a ResNet-50.
- **PAN** ([ICCV'19](https://arxiv.org/abs/1908.05900)) has three parts:
  - A cheap neck: FPEM is a cascaded, depthwise U-shape, and FFM fuses its outputs.
  - Two predicted maps, text region and kernel, plus a 4-dim similarity embedding.
  - A loss with discriminative terms Lagg/Ldis that pull pixels toward their own kernel and push kernels apart ([pdf](https://arxiv.org/pdf/1908.05900), Eq. 1–5).
- **DBNet** ([AAAI'20](https://arxiv.org/abs/1911.08947)):
  - **Head:** probability map P, threshold map T, and approximate binary map B̂ = σ(k(P−T)) with k=50 ([DBNet++ pdf](https://arxiv.org/pdf/2202.10304)).
  - **Labels:** the shrink offset is D = A(1−r²)/L with r=0.4 (A = area, L = perimeter). The threshold label is the distance-to-edge map inside the dilated polygon.
  - **Inference:** binarize P at 0.2, take connected components, then unclip by D′ = A′·1.5/L′. The T branch is dropped, so DB adds **zero inference cost** ([pdf](https://arxiv.org/pdf/1911.08947)).
  - **Gains:** DB alone adds +3.7 F (R18, MSRA-TD500) and +4.9 F (R18, CTW1500) at the same FPS (Tab. 1).
- **DBNet++** ([TPAMI](https://arxiv.org/abs/2202.10304)) adds **ASF**: concat 4 scales → 3×3 conv → spatial attention → N per-scale weights → weighted concat. ASF costs a few FPS. Example on Total-Text: R18 82.8 F @ 50 FPS (DB) vs 83.3 F @ 48 FPS (DB++) (both [pdfs](https://arxiv.org/pdf/2202.10304)).
- **FAST** ([arXiv 2111.02394](https://arxiv.org/abs/2111.02394)) is the most aggressive real-time design:
  - **1-channel kernel output (MKR):** the kernel label is an erosion of the filled text region. Inference is CCL followed by **dilation implemented as `F.max_pool2d(kernel, s, 1, s//2)`**, so it is GPU/NPU-friendly with no Vatti clipping.
  - **TextNet backbone:** NAS over {3×3, 1×3, 3×1, identity}. The reparameterizable convs are merged into plain convs at inference. The asymmetric 1×3/3×1 kernels are chosen for extreme-aspect-ratio text.
  - **Results:** text dilation instead of PA gives +13.9 FPS at −0.2 F (Total-Text, R18). TextNet-B beats MobileNetV3, EfficientNet-B0 and OFANet as detection backbones at matched speed ([pdf](https://arxiv.org/pdf/2111.02394) Tab. I, III).
- **CentripetalText** ([NeurIPS'21](https://arxiv.org/abs/2107.05945)) uses kernels plus a 2-channel shift field (each pixel points toward its kernel) with a Relaxed L1 loss. CT-320 reaches ~93 FPS on Total-Text (no pretraining: 79.4 F; with: 81.2 F).
- **PP-OCR industrial line:**
  - **v2:** CML.
  - **v3:** LK-PAN teacher, RSE-FPN student.
  - **v4:** PFHead, Dynamic Shrink Ratio, PP-LCNetV3, KL added to CML.
  - **v5:** data-centric; the detector is essentially v4 ([2603.24373](https://arxiv.org/html/2603.24373v1)).
  - **v6 (2026)** is the newest real-time DB variant I found ([2606.13108](https://arxiv.org/html/2606.13108)):
    - Reparameterized LCNetV4 backbone.
    - RepLKFPN neck: the neck shrinks from 172K to 118K params after reparameterization, and the receptive field grows from 3×3 to 7×7.
    - Training-only auxiliary DB heads at P2/P3/P4, weighted 0.4/0.3/0.2.
    - Dice+Focal loss on the shrink map.
    - Tiny det backbone is 0.43M params.
    - It reports better resolution robustness than v5. At 2.83× scale, Medium reaches 87.94 Hmean vs 65.81 for v5-mobile (Tab. 5).
- **Other 2023–2026 real-time segmentation detectors:** I found no academic method in my searches that clearly supersedes FAST and DB++ on the speed/accuracy frontier. EdgeText and LRANet are regression/contour methods. RSCA (2021, [arXiv 2105.12789](https://arxiv.org/abs/2105.12789)) is another real-time segmentation method; I did not read it.

## 3. Regression / contour / transformer detectors (Item 2)

- **EAST** ([CVPR'17](https://arxiv.org/abs/1704.03155)) is a dense per-pixel quad/rotated-box regressor with NMS: 78.2 F on IC15 at 13.2 FPS. It is simple to export, but it handles curved text poorly and needs NMS (locality-aware NMS `[UNVERIFIED detail]`).
- **FCENet** ([CVPR'21](https://arxiv.org/abs/2104.10442)) represents contours with a Fourier signature (K=5 by default `[UNVERIFIED]`; the paper fixes N=50 sample points). It is strong on highly curved text (85.5–85.8 F). Its post-processing is IFT + NMS.
- **TextBPN++** ([T-MM, arXiv 2205.05320](https://arxiv.org/abs/2205.05320)) refines boundary points iteratively with a transformer. It is accurate but ~13 FPS on a desktop GPU (repo re-impl, `[UNVERIFIED]`).
- **DPText-DETR** ([AAAI'23](https://arxiv.org/abs/2207.04491)), **DeepSolo** ([CVPR'23](https://arxiv.org/abs/2211.10772)) and **SRFormer** ([AAAI'24](https://arxiv.org/abs/2308.10531)) all run ResNet-50 plus a deformable-DETR decoder. They hold the curved-text F records: Total-Text 89.0 (DPText-DETR) and 90.0 (SRFormer) per the repos. None reports mobile latency, and the repos give no FPS.
- **LRANet** ([AAAI'24](https://arxiv.org/abs/2306.15142)) uses a low-rank contour basis. It reports ~37 FPS on a 3090 with high F (secondary summary, `[UNVERIFIED]`). LRANet++ ([2511.05818](https://arxiv.org/abs/2511.05818), Nov 2025) extends it to spotting.
- **Real-time DETR family → text?**
  - The family: RT-DETRv4 ([2510.25257](https://arxiv.org/abs/2510.25257)), D-FINE ([2410.13842](https://arxiv.org/abs/2410.13842)), DEIMv2 ([2509.20787](https://arxiv.org/abs/2509.20787)) and EdgeCrafter/ECDet ([2603.18739](https://arxiv.org/abs/2603.18739)).
  - **I found no primary-source paper adapting these to polygon text detection** (searched Nov 2026).
  - The closest bridges are point-query text DETRs (DPText-DETR, SRFormer), which are heavy.
  - The real-time DETRs are measured on **T4 + TensorRT FP16** (e.g., D-FINE-L 54.0 AP @ 124 FPS; RT-DETRv4 49.7–57.0 AP @ 273–78 FPS; ECDet-S 5.41 ms). They report no ANE or mobile-NPU numbers. EdgeCrafter itself notes ViTs are less well optimized on edge hardware ([html](https://arxiv.org/html/2603.18739)).
- **Mobile NPU viability (my assessment, not from a source):**
  - DETR decoders bring deformable attention (grid-sample), top-k query selection and Hungarian-trained heads. These are typically poorly supported or fall back to CPU/GPU on ANE, MNN-NPU and NNAPI `[UNVERIFIED]`.
  - Dense parcel labels can have 100+ words, which stresses query budgets (300 queries in ECDet).
  - **Verdict:** segmentation + DB remains the pragmatic choice for ANE/MNN. The real-time DETRs' *training recipes* are the transferable part (VFM feature distillation, deep supervision, O2O/dense matching); the decoders are not.

## 4. Loss-function modifications (Item 3)

**DB loss as published vs. as implemented**
- *Paper:* BCE with OHEM (pos:neg = 1:3) on both Ls and Lb, L1 on the threshold map, with α=1, β=10 ([pdf](https://arxiv.org/pdf/1911.08947), Eq. 7–9).
- *PaddleOCR `DBLoss`* differs: OHEM-balanced BCE on the shrink map ×α=5, **Dice** on the binary map ×1, MaskL1 on the threshold map ×β=10 ([source](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/losses/det_db_loss.py)). PP-OCRv2 also describes the GT loss as BCE + Dice + L1 ([pdf](https://arxiv.org/pdf/2109.03144) Eq. 1).
- **Check which variant our trainer uses.** The weights are not interchangeable between the two forms.

**Dice / IoU / focal**
- PSENet, PAN and FAST all use Dice, with OHEM on the text-region term.
- PP-OCRv6 replaces the shrink-map loss with **Dice + Focal (α=0.25, γ=2.5)**. It applies the same to aux heads and keeps L1 on the threshold map ([html](https://arxiv.org/html/2606.13108) Eq. 3).
- No ablation isolating Focal vs BCE-OHEM was available to me `[UNVERIFIED gain]`.

**Shrink map vs kernel representations**
- DB: one shrink map plus a learned threshold; unclip on CPU.
- PSE/PAN: multi-kernel or kernel + embedding.
- FAST: one eroded kernel, dilated back by max-pool.
- CT: kernel + shift field.
- FAST's ablation: MKR + dilation costs −0.2 F but gives +13.9 FPS vs PA ([pdf](https://arxiv.org/pdf/2111.02394) Tab. I).

**Adaptive shrink / scale**
- PP-OCRv4's **Dynamic Shrink Ratio** ramps r linearly from 0.4 to 0.6 over training, giving +1.27 hmean on the student (76.97→78.24) ([docs](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)).
- DBNet++ ASF learns scale weights at the feature level.

**Instance-balanced / small-text weighting**
- PixelLink-style instance-balanced CE weights each instance's pixels inversely to its area ([AAAI'18](https://arxiv.org/abs/1801.01315), weighting detail `[UNVERIFIED]`, not confirmed on the abstract page).
- I found no recent real-time text paper with an ablated small-text weighting. **Open opportunity.**

**Boundary / regression losses**
- CT Relaxed L1 for shift fields ([pdf](https://arxiv.org/pdf/2107.05945)).
- EdgeText PI-loss for curve parameters ([abs](https://arxiv.org/abs/2504.04001)).
- FCENet Lreg in Fourier space: the regression loss alone is worth +6.9 / +9.3 hmean on CTW1500 / Total-Text ([pdf](https://arxiv.org/pdf/2104.10442)).

**Deep supervision**
- PP-OCRv6 attaches training-only aux DB heads to P2/P3/P4 (0.4/0.3/0.2). They are free at inference ([html](https://arxiv.org/html/2606.13108)).

**Distillation losses for detectors**
- **CML** (PP-OCRv2, [pdf](https://arxiv.org/pdf/2109.03144) Eq. 2–4): Loss_total = Loss_gt + Loss_dml + Loss_distill.
  - Loss_dml is a symmetric KL between the two students' probability maps.
  - Loss_distill = γ·BCE(S, dilate(T)) + Dice(S, dilate(T)), with γ=5 and a 2×2 dilation kernel.
  - The PaddleOCR `DistillationDilaDBLoss` thresholds the teacher shrink map at **0.3** before dilating ([source](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/losses/distillation_loss.py)).
- **PP-OCRv4** adds a KL-divergence loss between student and teacher response maps: 79.08→79.56 hmean per the text, 79.87 per the table ([docs](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html), internal inconsistency noted).
- **Channel-wise KD** (CWD, [ICCV'21](https://arxiv.org/abs/2011.13256)): spatial softmax per channel, then KL. It was designed for dense prediction and is a good fit for P/T maps or neck features.
- **Focal & Global KD** (FGD, [CVPR'22](https://arxiv.org/abs/2111.11837)) separates foreground and background feature distillation, plus a global relation term. It gives +2.9 to +3.6 mAP on COCO detectors. Using the text mask as the foreground mask is a direct analogue.
- **VFM feature distillation in real-time detectors:**
  - **RT-DETRv4 DSI** ([html](https://arxiv.org/html/2510.25257v1)):
    - Teacher: frozen DINOv3-ViT-B.
    - Aligns **only the deepest encoder feature (F5)** through a **linear projector** with **negative cosine similarity**. Cosine beat MSE: 53.5 vs 52.7 AP.
    - Aligning backbone S3/S4/S5 gave no gain.
    - λ is set adaptively by **GAM**, which drives the AIFI share of gradient-norm toward a target ratio. GAM reached 55.4 AP vs 55.1 for the best static λ=20.
  - **EdgeCrafter** ([html](https://arxiv.org/html/2603.18739)):
    - Teachers are DINOv3-S/B *first adapted to detection* (ECTeacher).
    - A linear adapter does squared-L2 alignment of the student's final features to the teacher's last two blocks.
    - The DINOv3-L teacher was *worse* than B (52.6 vs 54.3 AP).
  - **DEIMv2** ([abs](https://arxiv.org/abs/2509.20787)) uses DINOv3(-distilled) backbones with a Spatial Tuning Adapter for S–X. Its sub-2M variants (Pico 1.5M, 38.5 AP) use pruned HGNetv2, not DINOv3.

## 5. Backbones for edge detectors (Item 4)

| Backbone | Size / acc. | On-device evidence | Notes |
|---|---|---|---|
| MobileNetV3 | — | PP-OCRv2/v3 student ([pdf](https://arxiv.org/pdf/2109.03144)) | Baseline; FAST shows TextNet beats it for detection ([pdf](https://arxiv.org/pdf/2111.02394)) |
| PP-LCNetV3 | det model 4.8M | 67–94 ms CPU (Intel) in PP-OCRv4 ([docs](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)) | PP-OCRv4/v5 mobile backbone |
| LCNetV4 (PP-OCRv6) | det bb 0.43/2.48/15.5M | V100 / Xeon only; end-to-end 0.35 s on Apple M4 via ONNX Runtime (incl. rec) ([html](https://arxiv.org/html/2606.13108)) | Reparameterized multi-branch DW |
| ConvNeXt V2 A/F/P | 3.7M/5.2M/9.1M; 76.7/78.5/80.3 top-1 ([repo](https://github.com/facebookresearch/ConvNeXt-V2)) | No official mobile latency found | **ImageNet weights CC-BY-NC** ([repo](https://github.com/facebookresearch/ConvNeXt-V2)), a licensing check for production. GRN and LayerNorm on ANE/fp16: `[UNVERIFIED]` perf/precision risk; benchmark it |
| FastViT | T8 76.2% @ 0.8 ms; SA12 80.9% @ 1.6 ms, **iPhone 12 Pro** ([repo](https://github.com/apple/ml-fastvit)) | Apple ModelBench | RepMixer reparam; "1.9× faster than ConvNeXt on mobile at same accuracy" ([abs](https://arxiv.org/abs/2303.14189)) |
| MobileOne | <1 ms, 75.9% on iPhone12 ([abs](https://arxiv.org/abs/2206.04040)) | Apple | Reparameterized, plain at inference |
| RepViT | >80% top-1 @ 1.0 ms on iPhone 12 ([abs](https://arxiv.org/abs/2307.09283)) | iPhone 12 (CoreML `[UNVERIFIED]`) | MobileNetV3 + ViT macro design, pure CNN |
| MobileNetV4 | "mostly Pareto optimal" incl. Apple Neural Engine; Hybrid-L 87% @ 3.8 ms Pixel 8 EdgeTPU ([abs](https://arxiv.org/abs/2404.10518)) | ANE named but per-device numbers not in the abstract | UIB block; Mobile MQA |
| EfficientViT | — | `[UNVERIFIED]`; not checked this session | Linear attention; may not map well to ANE |
| TextNet-T/S/B | 6.8/8.0/8.9M ([pdf](https://arxiv.org/pdf/2111.02394)) | GPU only | Text-specific NAS with 1×3/3×1 reparam convs |
| DINOv3 ConvNeXt T/S/B/L | 29/50/89/198M ([repo](https://github.com/facebookresearch/dinov3)) | — | Distilled from DINOv3 ViT-7B `[UNVERIFIED wording; repo says ConvNeXt distillation code released]`; DINOv3 License |

**General ANE guidance** ([Apple ML](https://machinelearning.apple.com/research/neural-engine-transformers)):
- Use a 4D channels-first layout.
- Avoid reshapes and transposes.
- Use Conv2d instead of Linear.
- Chunk large tensors.
- Pure-conv, reparameterized, BN-based nets (MobileOne/FastViT/RepViT) are the Apple-validated path.

**CoreML input shapes:** use `EnumeratedShapes` for best performance, up to 128 shapes. Flexible-shape models on the ANE need the "Infrequent" reshape-frequency hint on iOS 17.4+ ([coremltools](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)).

## 6. Open problems (Item 5)

1. **Dense small text.** DB's shrink (r=0.4) can erase tiny or thin words.
   - FAST keeps at least a minimal kernel per instance.
   - PP-OCRv4's DSR ramps r upward over training.
   - PP-OCRv6's resolution study shows mobile models lose accuracy sharply at both small and large scales.
   - No real-time paper I found ablates small-instance loss weighting.
2. **Long text lines / extreme aspect ratios.** Large receptive fields help:
   - LK-PAN 9×9: teacher +1.8 hmean ([docs](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv3_introduction.html)).
   - RepLKFPN 7×7 ([html](https://arxiv.org/html/2606.13108)).
   - FAST's 1×3/3×1 candidates.
   - DB's FPN at 1/4 resolution with 640-px inputs can still fragment long barcode-adjacent lines `[UNVERIFIED for our domain]`.
3. **Curved text.** Contour and DETR methods win on curved benchmarks (F 89–90) but are not mobile-viable as published. Parcel labels are mostly straight or perspective-warped, so this matters less for us.
4. **Post-processing cost on device.** DB needs threshold → findContours → minAreaRect/polygon → pyclipper unclip, all on CPU and scaling with the number of boxes.
   - FAST shows dilation can be a max-pool *inside* the graph ([pdf](https://arxiv.org/pdf/2111.02394) Alg. 1).
   - Only connected-component labeling remains outside the graph.
5. **Fixed-shape inputs.**
   - CoreML performs best with enumerated or fixed shapes.
   - Letterboxing to a fixed size wastes compute and shrinks text.
   - Multi-shape enumeration (e.g., portrait and landscape buckets) is supported up to 128 shapes ([coremltools](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)).
6. **Teacher–student scale gap.** Larger VFM teachers do not always help: DINOv3-L < DINOv3-B in EdgeCrafter. Adapting the teacher to the task first (ECTeacher) matters ([html](https://arxiv.org/html/2603.18739)).

---

## 7. Actionable ideas for our DB + ConvNeXt-V2 femto/pico student (DINOv3-ConvNeXt-T teacher)

Ordered roughly by expected value / effort. Items marked (src) have direct literature support; (hyp) are hypotheses to A/B test.

1. **Use cosine similarity, not MSE, for feature distillation, and pick the layer deliberately (src: [RT-DETRv4](https://arxiv.org/html/2510.25257v1)).**
   - Align the student's deepest neck or fused feature, not raw backbone stages, to the teacher through a linear (1×1) projector.
   - In RT-DETRv4, cosine beat MSE (53.5 vs 52.7 AP) and backbone-stage alignment gave no gain.
   - For a DB FPN, the analogous target is the fused 1/4-scale feature before the head, or the 1/32 feature. Ablate both.
2. **Fine-tune the teacher on parcel labels before distilling (src: [EdgeCrafter](https://arxiv.org/html/2603.18739) ECTeacher; PP-OCRv3 trained a strong DB-R50-LK-PAN-DML teacher, [docs](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv3_introduction.html)).**
   - A frozen generic DINOv3 is a weaker target than a detection-adapted one.
   - Don't jump to a bigger teacher: B beat L in EdgeCrafter.
3. **Add output-level map distillation alongside feature KD (src: [PP-OCRv2 CML](https://arxiv.org/pdf/2109.03144), [PP-OCRv4](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html)).**
   - BCE + Dice of the student's shrink and binary maps against the teacher's *thresholded (0.3) and 2×2-dilated* shrink map, γ=5.
   - Optionally a KL term on the response maps.
   - This needs a teacher with a DB head, which our setup has.
4. **Foreground-weighted feature KD (src: [FGD](https://arxiv.org/abs/2111.11837), [CWD](https://arxiv.org/abs/2011.13256)).**
   - Weight the feature-distillation loss by the GT text mask (dilated), or use channel-wise softmax-KL.
   - Parcel images are mostly background, so uniform MSE wastes capacity.
5. **Training-only auxiliary DB heads on P3/P4 (src: [PP-OCRv6](https://arxiv.org/html/2606.13108), weights 0.4/0.3/0.2).** They have zero inference cost and suit a tiny student.
6. **Try Dice + Focal (α=0.25, γ=2.5) on the shrink map (src: [PP-OCRv6](https://arxiv.org/html/2606.13108)).**
   - Also try a **Dynamic Shrink Ratio** of 0.4→0.6 (src: [PP-OCRv4](https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html), +1.27 hmean on student). A larger final r keeps small dense label text from vanishing.
   - Re-tune the unclip ratio when r changes.
7. **Confirm our DB loss variant.** Check paper BCE-both (α=1) vs PaddleOCR (α=5 on shrink BCE, Dice on binary). Ported Paddle configs or weights assume the latter ([source](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/losses/det_db_loss.py)).
8. **Use a large-kernel / reparameterized neck (src: LK-PAN 9×9, RepLKFPN 7×7 DW).**
   - ConvNeXt's 7×7 DW already gives backbone receptive field.
   - A DW large-kernel FPN refinement block is cheap and helps long tracking-number lines (hyp for our domain).
9. **Move post-processing into the graph where possible (src: [FAST](https://arxiv.org/pdf/2111.02394)).**
   - Binarize (and optionally max-pool dilate) inside the CoreML/MNN graph and output a uint8 mask.
   - Keep only contours/CC on CPU.
   - A FAST-style eroded-kernel + max-pool head is a candidate replacement for Vatti unclip (hyp: test accuracy on boxy labels).
10. **Benchmark ConvNeXt-V2 femto/pico on ANE before investing more (hyp).**
    - GRN (global spatial L2 norm) and LayerNorm may be costly or fp16-sensitive on ANE `[UNVERIFIED]`.
    - Compare against a reparameterized student of similar FLOPs (RepViT / MobileOne / FastViT-T8/T12) with the same teacher.
    - **License:** ConvNeXt-V2 ImageNet weights are CC-BY-NC ([repo](https://github.com/facebookresearch/ConvNeXt-V2)), so check whether production uses them as init.
11. **Fixed-shape export: use CoreML `EnumeratedShapes`** (e.g., 2–4 aspect buckets) rather than one letterboxed square. Check the ANE compile for every shape ([coremltools](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)).
12. **Adaptive KD weight (src: RT-DETRv4 GAM).** If the KD loss swamps or vanishes relative to the DB loss across training, set λ from the gradient-norm share rather than hand-tuning.

---

## 8. Sources

Primary (papers, official repos and docs):
- EAST — https://arxiv.org/abs/1704.03155
- PixelLink — https://arxiv.org/abs/1801.01315
- PSENet — https://arxiv.org/abs/1903.12473 (pdf read)
- PAN — https://arxiv.org/abs/1908.05900 (pdf read)
- DBNet — https://arxiv.org/abs/1911.08947 (pdf read)
- DBNet++ — https://arxiv.org/abs/2202.10304 (pdf read)
- PAN++ — https://arxiv.org/abs/2105.00405
- FCENet — https://arxiv.org/abs/2104.10442 (pdf read)
- CentripetalText — https://arxiv.org/abs/2107.05945 (pdf read)
- FAST — https://arxiv.org/abs/2111.02394 (pdf read)
- TextBPN++ — https://arxiv.org/abs/2205.05320
- DPText-DETR — https://arxiv.org/abs/2207.04491 ; https://github.com/ymy-k/DPText-DETR
- DeepSolo — https://arxiv.org/abs/2211.10772
- SRFormer — https://arxiv.org/abs/2308.10531 ; https://github.com/retsuh-bqw/SRFormer-Text-Det
- LRANet — https://arxiv.org/abs/2306.15142 ; LRANet++ — https://arxiv.org/abs/2511.05818
- EdgeText — https://arxiv.org/abs/2504.04001
- RSCA — https://arxiv.org/abs/2105.12789 (not read)
- TextDS — https://arxiv.org/abs/2606.28077
- PP-OCRv2 — https://arxiv.org/abs/2109.03144 (pdf read)
- PP-OCRv3 — https://arxiv.org/abs/2206.03001 ; https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv3_introduction.html
- PP-OCRv4 — https://www.paddleocr.ai/latest/en/version2.x/ppocr/blog/PP-OCRv4_introduction.html
- PaddleOCR 3.0 report — https://arxiv.org/abs/2507.05595
- PP-OCRv5 — https://arxiv.org/html/2603.24373v1
- PP-OCRv6 — https://arxiv.org/html/2606.13108
- PaddleOCR DB loss — https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/losses/det_db_loss.py
- PaddleOCR distillation losses — https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/losses/distillation_loss.py
- RT-DETRv4 — https://arxiv.org/abs/2510.25257 ; https://arxiv.org/html/2510.25257v1
- DEIMv2 — https://arxiv.org/abs/2509.20787
- D-FINE — https://arxiv.org/abs/2410.13842
- EdgeCrafter — https://arxiv.org/abs/2603.18739 ; https://arxiv.org/html/2603.18739
- DINOv3 — https://arxiv.org/abs/2508.10104 ; https://github.com/facebookresearch/dinov3
- ConvNeXt V2 — https://github.com/facebookresearch/ConvNeXt-V2
- FastViT — https://arxiv.org/abs/2303.14189 ; https://github.com/apple/ml-fastvit
- MobileOne — https://arxiv.org/abs/2206.04040
- RepViT — https://arxiv.org/abs/2307.09283
- MobileNetV4 — https://arxiv.org/abs/2404.10518
- Channel-wise KD — https://arxiv.org/abs/2011.13256
- FGD — https://arxiv.org/abs/2111.11837
- Apple ANE Transformers — https://machinelearning.apple.com/research/neural-engine-transformers
- coremltools flexible inputs — https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html
