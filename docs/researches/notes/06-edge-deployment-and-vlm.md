# 06 — Edge deployment & efficiency, and the VLM / end-to-end document-model wave

Track: (A) edge deployment & efficiency for OCR; (B) VLM / document foundation models and what they mean for on-device OCR.
Period: 2022 → Nov 2026. Compiled 2026-11-10.

Conventions:
- Numbers carry an inline link to where they were read. Items tagged `[UNVERIFIED]` were **not** confirmed from a primary source during this pass (memory, secondary blog, or my own analysis) and must be checked before being relied on.
- "Analysis" = my engineering reasoning, not a published result.

---

## Part A — Efficiency

### 1. Hardware-aware backbone design relevant to OCR

**Key lesson, repeated across vendors: FLOPs/params are poor latency proxies.** MobileOne says FLOPs and parameter count "may not correlate well with latency of the network when deployed on a mobile device" ([MobileOne](https://arxiv.org/abs/2206.04040)). EfficientViT-MSRA says transformer speed is "commonly bounded by memory inefficient operations" such as reshapes and element-wise functions in MHSA ([EfficientViT-MSRA](https://arxiv.org/abs/2305.07027)).

| Family | Core idea | Published mobile numbers | OCR relevance |
|---|---|---|---|
| **PP-LCNet (v1)** | CPU/MKLDNN-oriented. H-Swish, SE **only in the last two blocks** because SE is slow on Intel CPU, 5×5 DW conv only at the tail, 1280-d 1×1 conv after GAP ([PP-LCNet HTML](https://arxiv.org/html/2109.15099v1)) | PP-LCNet-1x: 3.0M params, 161M FLOPs, 71.32% top-1 (74.39% with SSLD), 2.46 ms on Xeon Gold 6148 with MKLDNN, 10 threads ([PP-LCNet](https://arxiv.org/html/2109.15099v1)) | Backbone of the PP-OCRv3/v4/v5 mobile det/rec models. Using LCNetV3 in mobile v4/v5 is `[UNVERIFIED]`: the PaddleOCR 3.0 report does not name the mobile backbone ([PaddleOCR 3.0](https://arxiv.org/html/2507.05595v1)) |
| **PP-LCNetV4 / "LCNetV4"** (PP-OCRv6, Jun 2026) | MetaFormer split: 3×3 DW token mixer plus pointwise expand/compress channel mixer (ratio 2, **GELU**), **BatchNorm**, optional SE, **re-parameterised DW branches** (3×3 + 1×1 + identity). Same backbone for det and rec through stride configs ([PP-OCRv6](https://arxiv.org/html/2606.13108)) | See §5 | Directly relevant to our PP-OCRv6 det path. GELU is the main new op versus LCNetV3-era HardSwish (analysis) |
| **RepVGG / MobileOne / FastViT / RepViT** (structural re-param) | Multi-branch at train time, folded into one conv at inference | MobileOne: under 1 ms on iPhone 12 at 75.9% top-1 ([MobileOne](https://arxiv.org/abs/2206.04040)). RepViT: over 80% top-1 at 1.0 ms on iPhone 12 ([RepViT](https://arxiv.org/abs/2307.09283)). FastViT: 3.5× faster than CMT, 4.9× faster than EfficientNet, 1.9× faster than ConvNeXt on mobile at equal accuracy ([FastViT](https://arxiv.org/abs/2303.14189)) | Apple measures these on iPhone/ANE, so they are the closest proven ANE-friendly families. Re-param costs nothing at inference, so it suits detection necks too (PP-OCRv6 uses RepLKFPN with 7×7 DW plus dilated branches ([PP-OCRv6](https://arxiv.org/html/2606.13108))) |
| **MobileNetV4** | UIB block merges IB, ConvNeXt-style, FFN and ExtraDW; **Mobile MQA** attention gives a "39% speedup"; "mostly Pareto optimal" across mobile CPU, DSP, GPU, **Apple Neural Engine** and EdgeTPU ([MNv4](https://arxiv.org/abs/2404.10518)) | MNv4-Hybrid-L: 87% ImageNet top-1 at 3.8 ms on Pixel 8 EdgeTPU ([MNv4](https://arxiv.org/abs/2404.10518)) | Best-documented "universal" option if one backbone has to serve CoreML-ANE, QNN-HTP and CPU. Conv-only MNv4 variants avoid LayerNorm and attention entirely (analysis) |
| **EfficientViT (MIT)** | Multi-scale linear attention built only from "hardware-efficient operations"; contrasted with softmax attention and large-kernel conv ([EfficientViT-MIT](https://arxiv.org/abs/2205.14756)) | Abstract gives GPU numbers only | Linear attention suits high-res dense prediction such as detection. No published OCR use found |
| **EfficientViT (MSRA)** | Cascaded group attention; memory-bound ops dominate | M5 is +1.9% over MNv3-L with 40.4% / 45.2% higher throughput on V100 / Xeon ([EfficientViT-MSRA](https://arxiv.org/abs/2305.07027)) | — |
| **ConvNeXt-V2** atto/femto/pico | ConvNeXt plus **GRN** (Global Response Normalization), FCMAE pre-training ([ConvNeXt V2](https://arxiv.org/abs/2301.00808)) | Atto 3.7M / 76.7%; Femto 5.2M / 0.78 GFLOPs / 78.5%; Pico 9.1M / 1.37 GFLOPs / 80.3%; Nano 15.6M / 81.9% ([repo](https://github.com/facebookresearch/ConvNeXt-V2)) | Our det student. Uses **LayerNorm + GELU + 7×7 DW + GRN**, which is the least NPU-friendly op mix in this table (see §1a) |
| **SHViT** | Single-head attention, larger-stride patchify stem | S4 is 3.3× / 8.1× / 2.4× faster than MobileViTv2-1.0 on GPU / CPU / iPhone 12, with +1.3% accuracy ([SHViT](https://arxiv.org/abs/2401.16456)) | — |
| **StarNet** | "Star operation" (element-wise multiplication) ([StarNet](https://arxiv.org/abs/2403.19967)) | Abstract gives no numbers | Element-wise mul is cheap everywhere |
| **LSNet** | LS conv: large-kernel perception plus small-kernel aggregation ([LSNet](https://arxiv.org/abs/2503.23135)) | Abstract gives no numbers | — |

PP-OCR's own ablation shows that **SE is a poor trade on mobile CPU for detection**. Removing SE from the MobileNetV3 det backbone cut model size by 36.6% and inference time by 18.8% with no HMean loss (Table 3: 0.6046 to 0.6087, 213 ms to 173 ms) ([PP-OCR v1 paper, Table 3](https://arxiv.org/pdf/2009.09941)).

#### 1a. Ops that are slow or unsupported per runtime

**Apple Neural Engine / CoreML (primary guidance):**
- Use (B, C, 1, S) channels-first 4D layout and replace `nn.Linear` with `nn.Conv2d` 1×1. Avoid reshapes and transposes. Chunk Q/K/V per head and split softmax per head for L2 residency. The last axis must be contiguous and 64-byte aligned, and a singleton last axis inflates memory by **32× (fp16)**. DistilBERT ran "up to 10 times faster" with 14× lower peak memory on iPhone 13 ([Apple – Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers)).
- "ANE supports a maximum of 5D tensors", so window partition has to be rewritten as 5D. With one fp16 element on the last dimension, effective speed is "32x slower than the maximum allowed". "ANE runs convolution operations well" ([Apple – Deploying Attention-Based Vision Transformers to ANE](https://machinelearning.apple.com/research/vision-transformers)).
- Neither Apple article lists LayerNorm, GELU or einsum as unsupported. The first article *recommends* an einsum form (`bchq,bkhc->bkhq`) that maps directly to hardware ([Apple](https://machinelearning.apple.com/research/neural-engine-transformers)). Claims that "LayerNorm/GELU fall back to GPU/CPU on ANE" are therefore `[UNVERIFIED]` and should be checked per model with Xcode's Core ML performance report.
- Compression: palettization "typically works the best on the Neural Engine". W8A8 gives "considerable latency benefits on the Neural Engine" on newer hardware (A17 Pro, M4) with int8-int8 compute. Some backends may decompress weights so latency equals fp16 ([coremltools optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)).
- ExecuTorch Core ML backend: lowers to fp16 by default unless `compute_precision` says otherwise. Keep the PyTorch model in fp32 before lowering to avoid dtype-mismatch errors. The "CoreML7" opset requires iOS 17+ ([ExecuTorch Core ML troubleshooting](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-troubleshooting.html)).

**Qualcomm QNN / Hexagon HTP:**
- "The QNN HTP backend only supports quantized models" (via ONNX Runtime). An fp16 path exists via `enable_htp_fp16_precision`. "QNN EP does not support models with dynamic shapes". Typical types are uint8/uint16, and sensitive tensors can be promoted to 16-bit ([ORT QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)).
- ExecuTorch QnnQuantizer offers 8a8w (default), 16a16w, 16a8w, 16a4w and 16a4w_block, with PTQ or QAT, and fp16 via `use_fp16`. Verified SoCs: SM8550, SM8450 ([ExecuTorch Qualcomm](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)).
- QNN HTP op revision history lists GELU with INT16 activations (SDK 2.33) and "All" (2.43), and LayerNorm INT8/INT16 (2.34). Release notes mention a fixed "incorrect scale on quantized LayerNorm" accuracy bug and fixed "slower GELU inference cycles" ([QNN HTP op revision history](https://docs.qualcomm.com/bundle/publicresource/80-63442-10/topics/htp_opdef_version_history.md); seen via search snippet, `[UNVERIFIED]` exact wording). Takeaway: LN and GELU on HTP depend on SDK version, so pin and test the SDK.

**TFLite / LiteRT GPU delegate:**
- The documented fp16/fp32 op list includes CONV_2D, DEPTHWISE_CONV_2D, ADD, MUL, LOGISTIC, RELU/RELU6/PRELU, SOFTMAX, RESIZE_BILINEAR, TRANSPOSE_CONV and others. It does **not** list HARD_SWISH, GELU or LayerNorm. Partially delegated graphs "often" run slower than CPU-only. Quantized models execute as float on GPU ([LiteRT GPU](https://developers.google.com/edge/litert/performance/gpu)). The page may be outdated relative to the ML Drift / LiteRT-Next GPU backend: `[UNVERIFIED]`.

**MNN:**
- Options are 2–8-bit weight quantization (block size 32–128 recommended, HQQ option), offline int8 (KL with 100–1000 images, ADMM, or EMA, the last "may be more accurate"), dynamic quantization (`MNN_LOW_MEMORY`), fp16 storage (`--fp16`) and fp16 runtime (`Precision_Low`). FP16 is "usually <0.1% loss on vision models", but the docs warn about numerically sensitive tasks such as **small-object detection** ([MNN compression guide](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html)).

**ExecuTorch:**
- Current backends overview lists XNNPACK (CPU), Core ML (NPU/GPU/CPU), MLX (Apple GPU, experimental on iOS), Vulkan (Android GPU, "mature on Android"), Qualcomm, MediaTek, Samsung Exynos, Arm and others. Unsupported ops fall back to CPU, and partitioners can be stacked in priority order. **MPS is no longer listed** ([ExecuTorch backends](https://docs.pytorch.org/executorch/stable/backends-overview.html)). Its deprecation status is `[UNVERIFIED]`.
- Core ML quantizer: PT2E 8-bit static or weight-only, and "dynamic quantization is not supported by CoreML". 4-bit per-group and palettization (1–8 bit LUT) need iOS 18. PT2E quantization needs iOS 17+ ([ExecuTorch Core ML quantization](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-quantization.html)).

#### Op-compatibility table (op × runtime)

Legend: ✅ supported and generally fast · ⚠️ supported with caveats (slow, precision-sensitive, or version-dependent) · ❌ not supported / falls back · ? not established. Cells without a link are analysis or `[UNVERIFIED]`.

| Op / pattern | CoreML-ANE | CoreML-GPU | QNN HTP | MNN CPU (ARM) | LiteRT GPU delegate | XNNPACK (ExecuTorch/LiteRT CPU) |
|---|---|---|---|---|---|---|
| Conv / DW conv 3×3, BN folded | ✅ "ANE runs convolution operations well" ([Apple](https://machinelearning.apple.com/research/vision-transformers)) | ✅ | ✅ (quantized) | ✅ | ✅ ([LiteRT](https://developers.google.com/edge/litert/performance/gpu)) | ✅ |
| Large-kernel DW (7×7) | ✅/⚠️ FastViT/RepLKFPN show it is usable; cost grows with kernel size `[UNVERIFIED]` | ✅ | ⚠️ `[UNVERIFIED]` | ⚠️ memory-bound (analysis) | ✅ | ⚠️ |
| BatchNorm (inference) | ✅ folded into conv | ✅ | ✅ | ✅ | ✅ | ✅ |
| LayerNorm (channels-first, ConvNeXt style) | ? not listed as unsupported by Apple; check perf report `[UNVERIFIED]` | ✅ | ⚠️ SDK-version-dependent, past quantized-LN scale bug ([QNN](https://docs.qualcomm.com/bundle/publicresource/80-63442-10/topics/htp_opdef_version_history.md)) | ✅ `[UNVERIFIED]` | ❌ not in delegate op list ([LiteRT](https://developers.google.com/edge/litert/performance/gpu)) | ✅ |
| GELU | ? `[UNVERIFIED]` | ✅ | ⚠️ INT16 since 2.33, "All" since 2.43, GELU slowdown fixed in a later SDK ([QNN](https://docs.qualcomm.com/bundle/publicresource/80-63442-10/topics/htp_opdef_version_history.md)) | ✅ | ❌ not listed | ✅ |
| ReLU / ReLU6 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HardSwish / HardSigmoid (SE gate) | ✅ `[UNVERIFIED]` | ✅ | ✅ `[UNVERIFIED]` | ✅ | ❌ not in documented list | ✅ |
| SE (GAP → FC → gate → mul) | ⚠️ global pool plus broadcast mul is memory-bound; PP-LCNet avoids SE because it is slow on CPU ([PP-LCNet](https://arxiv.org/html/2109.15099v1)) | ⚠️ | ⚠️ | ⚠️ PP-OCR det: removing SE gave −18.8% latency on mobile CPU ([PP-OCR](https://arxiv.org/pdf/2009.09941)) | ✅ (pool+mul) | ⚠️ |
| GRN (ConvNeXt-V2) | ⚠️ **fp16 overflow risk** in the spatial L2 norm (analysis, see actionable #2) | ⚠️ same | ⚠️ needs 16-bit activations (analysis) | ✅ fp32 | ❌ needs reduce/sqrt/div (analysis) | ✅ |
| Softmax attention (MHSA) | ⚠️ needs (B,C,1,S) layout, per-head split, chunked QKV ([Apple](https://machinelearning.apple.com/research/neural-engine-transformers)) | ✅ | ⚠️ MatMul input type limits ([ORT QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)) | ✅ | ⚠️ | ✅ |
| Reshape/transpose-heavy blocks | ⚠️ "minimize memory copies" ([Apple](https://machinelearning.apple.com/research/neural-engine-transformers)) | ⚠️ | ⚠️ | ⚠️ memory-bound ([EfficientViT-MSRA](https://arxiv.org/abs/2305.07027)) | ⚠️ | ⚠️ |
| Tensors with rank > 5 | ❌ "maximum of 5D tensors" ([Apple](https://machinelearning.apple.com/research/vision-transformers)) | ? | ? | ✅ | ? | ✅ |
| einsum | ✅ in the specific form Apple recommends ([Apple](https://machinelearning.apple.com/research/neural-engine-transformers)); others lowered to matmul+transpose `[UNVERIFIED]` | ✅ | ⚠️ converter-dependent | ⚠️ | ❌ | ⚠️ |
| Dynamic input shapes | ⚠️ supported via enumerated/flexible shapes; ANE prefers fixed `[UNVERIFIED]` | ✅ | ❌ "does not support models with dynamic shapes" ([ORT QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)) | ✅ (resize session) | ⚠️ | ✅ |
| LSTM (old CRNN head) | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ "Basic LSTM only" ([LiteRT](https://developers.google.com/edge/litert/performance/gpu)) | ✅ |
| int8 W8A8 conv | ✅ fast path on A17 Pro / M4 ([coremltools](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)) | ⚠️ | ✅ required ([ORT QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)) | ✅ offline quant ([MNN](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html)) | ⚠️ runs as float ([LiteRT](https://developers.google.com/edge/litert/performance/gpu)) | ✅ |

### 2. Quantization for OCR

**Published OCR-specific evidence (PP-OCR v1, SD855 = Snapdragon 855) ([paper](https://arxiv.org/pdf/2009.09941)):**
- Standard PACT clips at ReLU semantics. MobileNetV3's **hard-swish produces negative activations, so ordinary PACT "leads to a higher quantization loss"**. PaddleOCR changed the PACT formula to clip at both ±α and added L2 decay of 0.001 on α.
- Direction classifier with PACT: 0.85 MB → 0.46 MB, 3.21 ms → 2.38 ms, accuracy 0.9403 → 0.9456 (Table 8).
- CRNN recognizer with PACT, **LSTM layers left unquantized**: 4.6 MB → 1.5 MB, 12 ms → 11 ms on SD855, accuracy 0.6581 → 0.674 (Table 13). Speedup was small because the LSTM stayed in float.
- Whole system with FPGM + PACT: 8.1 MB → 3.5 MB, 306 ms → 268 ms on SD855, F-score 0.5193 → 0.5210 (Table 14).
- Takeaways: (a) QAT/PACT can be *lossless or better* for small OCR models; (b) the sequence head (LSTM, or the SVTR neck in later versions) is the hard-to-quantize part; (c) quantization gains on mobile CPU are modest unless the whole graph is integer.

**Gaps and partial evidence:**
- No 2022–2026 primary study was found that measures int8 PTQ drops specifically for **CTC recognizers or DB detectors**. One search for such a study came up empty; the general PTQ literature is not OCR-specific.
- Commonly cited failure modes are all `[UNVERIFIED]` / analysis:
  - The large final FC/classifier of a CTC head (6k–18k classes) quantized per-tensor squeezes logits. Per-channel weights or 16-bit for the last layer is the usual fix.
  - DB probability and threshold maps after sigmoid lose resolution near the 0.3 binarization threshold, so keep the head output in fp16/16-bit.
  - SE hardsigmoid gates saturate.
  - Attention softmax and LayerNorm need 16-bit activations. This last point matches QNN's 16a8w/16a16w options and ORT's advice to promote "inaccurate tensor[s]" to 16-bit ([ORT QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)).
- MNN docs warn that fp16 or offline quant can hurt "small object detection" ([MNN](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html)), which is the relevant case for small text on parcel labels.

**VLM-side quantization evidence:** the LiteRT-LM PaddleOCR-VL-1.6 port keeps the **decoder in fp16** because "int4 and integer-compute int8 measurably corrupt transcription". Only the vision encoder and adapter are int8 ([litert-community HF](https://huggingface.co/litert-community/PaddleOCR-VL-1.6); community page, `[UNVERIFIED]` independent replication).

**Toolchain choices (primary docs):**
- CoreML: palettization (1–8-bit LUT), int8/int4 linear, W8A8, and pruning ([coremltools](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)).
- ExecuTorch-CoreML: PT2E static or weight-only, iOS 17+ ([ET-CoreML quant](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-quantization.html)).
- ExecuTorch-QNN: 8a8w / 16a8w / 16a4w with PTQ or QAT ([ET-QNN](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)).
- MNN: weight-only 2–8-bit, KL/ADMM/EMA offline int8, `auto_quant.py` with tolerance ([MNN](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html)).

### 3. Pruning, NAS, distillation

- **FPGM pruning (PP-OCR v1 det):** size −46.2%, SD855 latency −18.9% (164 → 133 ms), HMean 0.6239 → 0.6169 (Table 6) ([PP-OCR](https://arxiv.org/pdf/2009.09941)). Shrinking the det head's inner channels from 256 to 96 cut size by 41% and time by ~50% with a small HMean drop (Table 3). Shrinking the rec head from 256 to 48 channels took the model from 23M to 4.6M and made it ~30% faster with no accuracy loss (Table 11). In practice, **head width reduction beat pruning** in these tables.
- **NAS for OCR:** FAST searched a text-detection-specific backbone (TextNet), reaching 81.6% F at 152 FPS on Total-Text and over 600 FPS with TensorRT. It uses a 1-channel kernel representation plus GPU-parallel post-processing ([FAST](https://arxiv.org/abs/2111.02394)). MobileNetV4 is the main general-purpose NAS backbone with ANE/DSP targets ([MNv4](https://arxiv.org/abs/2404.10518)). No other OCR-specific NAS of note was found in 2023–2026 `[UNVERIFIED]` (not exhaustively searched).
- **Distillation is the dominant compression method in the PP-OCR line:**
  - PP-OCRv3 uses DML distillation for det and attention-guided CTC training plus UDML for rec; Hmean is "5% higher than PP-OCRv2 under comparable inference speed" ([PP-OCRv3](https://arxiv.org/abs/2206.03001)).
  - PP-OCRv5 distills the **det** student from **GOT-OCR2.0's visual encoder** (feature alignment), and the rec model from a GTC-NRTR branch ([PaddleOCR 3.0](https://arxiv.org/html/2507.05595v1)).
  - PP-OCRv6 distills rec tiny ← medium with CTC-logit KL (λ=1.0). It trains with an NRTR head only as a regularizer and adds auxiliary DB heads at P2–P4 during training only ([PP-OCRv6](https://arxiv.org/html/2606.13108)).

### 4. System-level tricks

- **Orientation stages are optional.** PaddleOCR 3.x marks doc-orientation (PP-LCNet_x1_0_doc_ori, 7 MB), UVDoc unwarping (30.3 MB, 869.82 ms CPU) and textline-orientation (PP-LCNet_x0_25, 0.96 MB) as "(Optional)" ([PaddleOCR OCR pipeline](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)). Unwarping is by far the most expensive. For parcel photos with a mostly-upright capture UI, dropping doc-ori and unwarp and keeping a 0.96 MB textline flip classifier is the low-cost option (analysis).
- **Fixed shapes:** QNN requires static shapes ([ORT QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)), and ANE prefers fixed shapes `[UNVERIFIED]`. For rec, use width buckets (e.g., 48×{160,320,640}) compiled as enumerated shapes or separate models, and pad crops to the next bucket (analysis). PP-OCR's rec input is 32×320 (v1), and the cls input moved from 32×100 to 48×192 for accuracy at ~equal speed (Table 8) ([PP-OCR](https://arxiv.org/pdf/2009.09941)).
- **Det resolution vs small-text recall:** PP-OCRv6 reports det timing at 2048 px (medium 106.89 ms GPU vs 253.52 ms for v5_server; CPU-ONNX 2327 ms vs 3035 ms) ([PP-OCRv6](https://arxiv.org/html/2606.13108)). Det cost scales roughly with pixel count, so tiling or ROI cropping (crop the label region first, then run full-res det inside it) is usually cheaper than raising the global det resolution (analysis).
- **Shared backbone:** PP-OCRv6 unifies det and rec on one backbone *family* (LCNetV4 with task-specific strides), but they are still two networks ([PP-OCRv6](https://arxiv.org/html/2606.13108)). True weight sharing between det and rec in production mobile OCR was not found `[UNVERIFIED]`.
- **Post-processing cost:** FAST moved text-line assembly to GPU-parallel ops because CPU post-processing was a bottleneck ([FAST](https://arxiv.org/abs/2111.02394)). For DB on phones, contour finding plus unclip runs on CPU, so its cost scales with probability-map size and box count (analysis; profile it).
- **Video/streaming:** No primary source found for OCR-specific caching or early exit on device `[UNVERIFIED]`. Standard practice: run det every N frames, track boxes, and re-recognize only crops that changed or are low-confidence (analysis).
- **Hybrid "read on device, escalate failures":** a blog benchmark simulation raised Android effective field-F1 from 75.5% to ~88.9% by escalating parse failures to a cloud VLM. Deterministic parsers alone raised a weak slice from 75.8% to 90.9% ([prasad.tech, Aug 2026](https://prasad.tech/blog/benchmarking-on-device-ocr/); single-author blog, not peer-reviewed).

### 5. Published on-device benchmarks

- **PP-OCR v1 on Snapdragon 855:** full system 3.5 MB, 268 ms ([PP-OCR](https://arxiv.org/pdf/2009.09941)).
- **PaddleOCR 3.x module latencies (T4 GPU / Xeon 6271C, 8 threads, not phones)** ([PaddleOCR docs](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)):
  - PP-OCRv5_mobile_det: Hmean 79.0, 4.7 MB, CPU 57.77 / 28.15 ms (standard / high-perf).
  - PP-OCRv5_mobile_rec: 81.29%, 16 MB, CPU 21.20 / 5.32 ms.
  - PP-OCRv4_mobile_det: 63.8, 4.7 MB.
  - PP-OCRv5_server_det: 83.8, 84.3 MB, CPU 383 ms.
  - PP-OCRv6 det tiny/small/medium: 80.6 / 84.1 / 86.2 Hmean at 1.9 / 9.6 / 59.4 MB. Rec tiny/small/medium: 73.5 / 81.3 / 83.2% at 4.4 / 20.4 / 73.3 MB. These are on an **internal** eval set and are "not directly comparable" with v5/v4 numbers.
- **PP-OCRv6 end-to-end s/image** ([PP-OCRv6 Table 9](https://arxiv.org/html/2606.13108)):
  - Apple M4, ONNX Runtime: tiny 0.35, small 1.29, medium 5.55, v5_mobile 1.10, v4_mobile 1.02.
  - Apple M4, PaddlePaddle: tiny 0.96 vs v5_mobile 5.82 (the "6.1×" headline).
  - Xeon 8350C, OpenVINO: tiny 0.20, v5_mobile 0.78.
  - **No Snapdragon/ARM phone numbers.**
  - Params (det/rec): tiny 0.43M/1.1M, small 2.48M/5.2M, medium 15.5M/19M.
  - Note: on ORT-M4, **v6_small is slower than v5_mobile** (1.29 vs 1.10 s).
- **Apple Vision / Live Text, Google ML Kit v2:** no official latency numbers published. ML Kit v2 covers Latin, Chinese, Devanagari, Japanese and Korean, and returns block/line/element/symbol levels with boxes, corner points, rotation and confidence ([ML Kit v2](https://developers.google.com/ml-kit/vision/text-recognition/v2)). Apple iOS 26 adds `RecognizeDocumentsRequest` (tables, lists, paragraphs, 26 languages) ([WWDC25-272](https://developer.apple.com/videos/play/wwdc2025/272/)). Whether it runs fully on-device is stated only by third parties `[UNVERIFIED]`.
- **Comparative (blog, Aug 2026):** field-level F1 on a synthetic badge/card/profile set was Apple Vision (iOS 26) 85.4%, ML Kit v2 75.5%, PP-OCRv4 74.7%, tesseract.js 59.0%; cloud Gemini 2.5 Flash reached 90.5% at p50 2.8 s. On-device latency was not measured ([prasad.tech](https://prasad.tech/blog/benchmarking-on-device-ocr/)). No 2024–2026 *peer-reviewed* on-phone engine comparison was found.

---

## Part B — VLM / document foundation models

### 6. Catalogue

OmniDocBench scores below come from three different versions (v1.0, v1.5, v1.6/1.7) and are **not comparable across versions**. Higher is better for "Overall" (v1.5+). v1.0 is reported as edit distance (lower is better).

| Model (date) | Params | Architecture | Resolution / token strategy | Reported scores | Boxes/coords? |
|---|---|---|---|---|---|
| Donut (2021/22) | ~200M `[UNVERIFIED]` | Swin encoder + BART decoder, OCR-free | fixed high-res | — | no |
| Nougat (2023) | 250M/350M `[UNVERIFIED]` | "Visual Transformer" → markup for scientific docs ([Nougat](https://arxiv.org/abs/2308.13418)) | fixed page | known repetition failure `[UNVERIFIED]` | no |
| Florence-2 (2023) | 0.23B / 0.77B `[UNVERIFIED]` | seq2seq unified prompts, FLD-5B with 5.4B annotations on 126M images ([Florence-2](https://arxiv.org/abs/2311.06242)) | 768² `[UNVERIFIED]` | — | OCR-with-region `[UNVERIFIED]` |
| GOT-OCR2.0 (Sep 2024) | 580M ([GOT](https://arxiv.org/abs/2409.01704)) | high-compression encoder + long-context decoder | dynamic res, multi-page | OmniDocBench v1.0 edit 0.349 ([PaddleOCR-VL tbl](https://arxiv.org/html/2510.14528)) | slice/box prompts |
| SmolDocling (Mar 2025) | 256M ([SmolDocling](https://arxiv.org/abs/2503.11576)) | DocTags markup | — | "competes with VLMs up to 27× larger" | DocTags with location |
| granite-docling-258M (Sep 2025) | 258M (Hub shows 0.3B) | Idefics3: SigLIP2-base-512 + Granite 165M LM ([HF](https://huggingface.co/ibm-granite/granite-docling-258M)) | pixel-shuffle | full-page OCR edit 0.45 (SmolDocling 0.48), TEDS 0.97, OCRBench 500 | `<loc_>` tokens. Card notes smaller models "may hallucinate more" |
| olmOCR (Feb 2025) / olmOCR-2 | 7B / 8B | fine-tuned 7B VLM; 260k-page training mix ([olmOCR](https://arxiv.org/abs/2502.18443)) | — | $176 per 1M pages vs $6,240 for GPT-4o ([olmOCR](https://arxiv.org/abs/2502.18443)); olmOCR-2 82.3 on olmOCR-Bench ([HF blog](https://huggingface.co/blog/ocr-open-models)) | grounding |
| MonkeyOCR (Jun 2025) | 3B; 0.6–1.2B via CPD ([MonkeyOCR](https://arxiv.org/abs/2506.05218)) | Structure-Recognition-Relation triplet | runs on one RTX 3090 | pro-3B 88.85 on OmniDocBench v1.5 ([PaddleOCR-VL tbl](https://arxiv.org/html/2510.14528)); V2 91.80 on v1.6 ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)) | layout boxes |
| dots.ocr (Jul 2025) → dots.mocr (Mar 2026) | "1.7B LLM", listed as 3B ([repo](https://github.com/rednote-hilab/dots.ocr)) | single VLM emits bbox + category + text JSON | — | 88.41 v1.5 ([PaddleOCR-VL tbl](https://arxiv.org/html/2510.14528)); 90.77 v1.6 ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)); 79.1 olmOCR-Bench. README says complex tables and formulas are still hard | **yes**, layout bboxes |
| MinerU2.5 (Sep 2025) / MinerU2.5-Pro | 1.2B ([MinerU2.5](https://arxiv.org/abs/2509.22186)) | **decoupled**: layout on a downsampled page, then recognition on native-res crops | coarse-to-fine | 90.67 v1.5; 93.04 v1.6, Pro 95.75 ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)); 1.0574 pages/s on A100 via vLLM ([PaddleOCR-VL](https://arxiv.org/html/2510.14528)) | layout boxes |
| PaddleOCR-VL (Oct 2025) | 0.9B: NaViT-style dynamic-res encoder + ERNIE-4.5-0.3B; layout by **PP-DocLayoutV2** (RT-DETR + 6-layer pointer network for reading order) ([paper](https://arxiv.org/html/2510.14528)) | max 2048×28×28 px, seq 16384 | 92.86 v1.5; v1.0 edit 0.115; 1.6184 pages/s on A100 (FastDeploy), +53.1% over MinerU2.5; 109 languages | layout boxes (from PP-DocLayoutV2) |
| PaddleOCR-VL-1.5 (Jan 2026) / 1.6 | 0.9B | adds seal recognition and **text spotting**; Real5-OmniDocBench robustness set ([1.5](https://arxiv.org/abs/2601.21957)) | — | 94.5 v1.5 ([1.5](https://arxiv.org/abs/2601.21957)); 1.6 scores 96.34 on v1.6 ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)) | spotting (format `[UNVERIFIED]`) |
| DeepSeek-OCR (Oct 2025) | DeepEncoder ~380M (SAM-base 80M + CLIP-L 300M, 16× conv compressor) + DeepSeek-3B-MoE-A570M ([paper](https://arxiv.org/html/2510.18234)) | **optical token compression**: Tiny 512²=64, Small 640²=100, Base 1024²=256, Large 1280²=400 tokens, Gundam tiling | 97% precision at <10× compression, ~60% at 20×; beats GOT (256 tok) with 100 tokens and MinerU2.0 (6000+) with <800; 200k+ pages/day on one A100 ([abs](https://arxiv.org/abs/2510.18234)); 75.4 olmOCR-Bench ([HF blog](https://huggingface.co/blog/ocr-open-models)) | grounding prompts `[UNVERIFIED]` |
| DeepSeek-OCR 2 (Jan 2026) | 3B ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)) | DeepEncoder V2 reorders visual tokens by semantics ("visual causal flow") ([abs](https://arxiv.org/abs/2601.20552)) | — | 90.25 v1.6 ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)) | — |
| HunyuanOCR (Nov 2025) | 1B: 0.4B native-res ViT + 0.5B Hunyuan LLM ([paper](https://arxiv.org/html/2511.19575v2)) | end-to-end, no layout model | arbitrary res, aspect preserved | 94.10 OmniDocBench (version unclear); OCRBench 860 (Qwen3VL-2B: 858); 89.95 on v1.6 board ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)) | **yes**: `<quad>` two-corner boxes normalized to [0,1000]; in-house spotting score 70.92 |
| LightOnOCR-1B (Oct 2025) / LightOnOCR-2 (2026) | ~1B ([arXiv 2601.14251](https://arxiv.org/abs/2601.14251)) | end-to-end, distillation mix; v2 adds image bbox via RLVR with IoU reward | — | 5.71 pages/s on H100, "5× faster than dots.ocr, 2× PaddleOCR-VL" (vendor) ([HF blog](https://huggingface.co/blog/lightonai/lightonocr)); SOTA olmOCR-Bench claim ([abs](https://arxiv.org/abs/2601.14251)) | images only |
| GLM-OCR (Mar 2026) | 0.9B: 0.4B CogViT + 0.5B GLM, **multi-token prediction**; layout by PP-DocLayout-V3 ([abs](https://arxiv.org/abs/2603.10910)) | layout, then parallel region recognition | — | 94.6 v1.5 (secondary source; `[UNVERIFIED]` from paper body) | layout boxes |
| Chandra (Datalab, 2025) / Chandra 2 | 9B (Qwen3-VL based) ([HF blog](https://huggingface.co/blog/ocr-open-models)); Chandra 2 "4B" `[UNVERIFIED]` | — | — | 83.1 olmOCR-Bench (model card via HF blog) | grounding |
| Qianfan-OCR (2026) | 4B | — | — | 93.90 v1.6 ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)) | — |
| Qwen2.5-VL-72B / Qwen3-VL-235B / InternVL3.5-241B | huge | general VLMs | — | 87.02 / — / 82.67 v1.5; Qwen3-VL-235B 89.78 v1.6 | weak boxes (see §8) |
| Newer (Sep 2026 additions) | — | TeleOCR, OvisOCR2, Unlimited-OCR added to the leaderboard ([OmniDocBench news 2026-09-11](https://github.com/opendatalab/OmniDocBench)) | — | not extracted | — |

Note: the OmniDocBench README says v1.7 shipped on 2026-04-30, but the visible leaderboard is labelled "v1.6_full". Version attribution for the v1.6 numbers above is therefore uncertain.

### 7. Trends and phone feasibility

- **Shrinking:** the leaders are now ~0.9–1.2B. PaddleOCR-VL, PaddleOCR-VL-1.5/1.6, GLM-OCR, HunyuanOCR, LightOnOCR and MinerU2.5 all beat 72B–235B generalists on OmniDocBench (tables above).
- **Decoupled layout + recognition wins on throughput and accuracy:** MinerU2.5 (coarse-to-fine), PaddleOCR-VL with PP-DocLayoutV2, and GLM-OCR with PP-DocLayout-V3 versus end-to-end (dots.ocr, HunyuanOCR, LightOnOCR). dots.ocr runs at 0.35 pages/s versus PaddleOCR-VL's 1.62 on an A100 ([PaddleOCR-VL](https://arxiv.org/html/2510.14528)).
- **Token compression:** DeepSeek-OCR uses 64–400 vision tokens per page ([paper](https://arxiv.org/html/2510.18234)). HunyuanOCR lists token compression as future work ([paper](https://arxiv.org/html/2511.19575v2)).
- **Phone feasibility (the only concrete data found):** PaddleOCR-VL-1.6 on LiteRT-LM is ~1.39 GB (int8 encoder/adapter, fp16 decoder) and uses a fixed **560×560** input (400 image tokens).
  - Galaxy S26 GPU: decode 35.6 tok/s, prefill 1239 tok/s, TTFT 190 ms, peak 1786 MB. The page is internally inconsistent about this run.
  - Galaxy S26 CPU: "did not run: degenerate_output".
  - Pixel 8a CPU: **22 s per page OCR**.
  - Source: [litert-community HF](https://huggingface.co/litert-community/PaddleOCR-VL-1.6), a community conversion, so all of these are `[UNVERIFIED]`.
  - Implication: a page with ~1k output tokens takes ~30 s even on a flagship GPU. That is fine as an offline or fallback path and impossible as a real-time scanner path (analysis).
  - The fixed 560² input and the extreme-aspect-ratio distortion warning are a mismatch for small text on parcel labels.
- **Known failure modes:**
  - Hallucination: PP-OCRv6 in-house "hallucination accuracy" is 93.20 for v6_medium versus 72.6–85.0 for frontier VLMs ([PP-OCRv6](https://arxiv.org/html/2606.13108)). The granite-docling card warns that small models hallucinate more ([HF](https://huggingface.co/ibm-granite/granite-docling-258M)).
  - Missing or poor boxes: frontier VLM detection Hmean is 12.0–46.8 versus 86.2 for v6_medium ([PP-OCRv6](https://arxiv.org/html/2606.13108)).
  - Repetition and degenerate output: S26 CPU run above.
  - Complex tables and formulas: dots.ocr README.
  - Language coverage: granite-docling CJK and Arabic support is "experimental"; olmOCR-2 is English-only ([HF blog](https://huggingface.co/blog/ocr-open-models)).
  - Latency: §7 phone numbers.

### 8. Coexistence thesis

**Classic det + CTC-rec still wins on:**
- **Latency and energy:** about 0.35 s per image on an M4 with ORT for PP-OCRv6-tiny ([PP-OCRv6](https://arxiv.org/html/2606.13108)), versus tens of seconds for a VLM page on a phone (above).
- **Size:** 1.5M params for v6 tiny end-to-end, versus ~1.4 GB for the smallest phone-ported doc VLM.
- **Boxes:** detection Hmean 86.2 for v6_medium versus ≤46.8 for GPT-5.5, Gemini-3.1-Pro and Qwen3-VL-235B on PaddlePaddle's in-house set ([PP-OCRv6](https://arxiv.org/html/2606.13108)). In-house benchmark from an interested party, so it needs independent confirmation.
- **Faithfulness:** CTC cannot invent text beyond its frame alignment (analysis). The PP-OCRv6 hallucination table points the same way.

**VLMs win on:** layout, reading order, tables, formulas, charts, multi-lingual long documents and semantic extraction (OmniDocBench). They also handle degraded or real-world photos in the newer models (PaddleOCR-VL-1.5's Real5-OmniDocBench).

**Hybrid evidence:**
1. **VLM as offline teacher or labeler for classic models (primary):** PP-OCRv5 used **ERNIE-4.5-VL-424B** for detection hard-case mining and to auto-annotate handwritten rec samples. It also used **GOT-OCR2.0's encoder** as the det distillation teacher ([PaddleOCR 3.0](https://arxiv.org/html/2507.05595v1)). DeepSeek-OCR is pitched as a training-data generator at 200k+ pages/day ([abs](https://arxiv.org/abs/2510.18234)).
2. **Classic layout model inside the VLM pipeline (primary):** PaddleOCR-VL uses PP-DocLayoutV2 and GLM-OCR uses PP-DocLayout-V3, so the "VLM" products are themselves hybrids.
3. **Fast path + VLM fallback:** blog-level evidence (Android 75.5% → ~88.9% simulated) ([prasad.tech](https://prasad.tech/blog/benchmarking-on-device-ocr/)). Peer-reviewed confirmation `[UNVERIFIED]`.
4. **LLM post-correction of small-model output:** not researched in depth this pass `[UNVERIFIED]`. The same blog shows deterministic parsers gave +15 F1 on one slice without any model.

---

## Actionable for our stack (DB det on ConvNeXt-V2 femto/pico or PP-LCNetV3/V4, PP-OCRv5-mobile-style CTC rec; CoreML / MNN / ExecuTorch)

1. **Profile the ConvNeXt-V2 student per op on ANE before investing further.** Its op mix (LayerNorm, GELU, 7×7 DW, GRN) is the least NPU-friendly in this survey. Use Xcode's Core ML performance report and check for CPU/GPU fallbacks. If LN/GRN fall back, consider a BN + ReLU/HardSwish re-param student (RepViT, MobileOne, MNv4-conv or LCNetV4 style), which Apple and Google benchmark on iPhone and ANE ([RepViT](https://arxiv.org/abs/2307.09283), [MobileOne](https://arxiv.org/abs/2206.04040), [MNv4](https://arxiv.org/abs/2404.10518)).
2. **GRN fp16 overflow check (analysis, high priority to verify).** GRN computes an L2 norm over H×W. On a detector feature map (e.g., 160×160 at stride 4 for 640 input), the sum of squares can exceed fp16 max (65504), giving inf/NaN or a collapsed GRN. CoreML lowers to fp16 by default ([ET-CoreML](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-troubleshooting.html)), and MNN `Precision_Low` and the GPU paths are also fp16. Fix options:
   - pre-scale before the norm (compute ‖x/s‖·s);
   - compute GRN in fp32 (op-level precision override);
   - or replace GRN in the student and re-distill.
   Compare fp32 and fp16 det maps on a few images to confirm.
3. **Fixed shapes everywhere.**
   - Det: fixed input (e.g., 640/736/960 square or letterboxed) per runtime. QNN HTP *requires* static shapes ([ORT QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)).
   - Rec: 2–3 width buckets (e.g., 48×320 and 48×640) as enumerated CoreML shapes or separate MNN sessions. Never use fully dynamic width on ANE or HTP.
4. **Quantization plan:**
   - Det: start with fp16 on ANE/GPU. If int8 is needed, keep the DB head output (prob/thresh maps) and any LN/GELU/GRN in 16-bit (QNN 16a8w; CoreML W8A8 only on A17 Pro/M4-class devices ([coremltools](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html))).
   - Rec: the large CTC classifier and the SVTR neck are the sensitive parts. Use per-channel weights and keep the final FC and attention in 16-bit or fp16.
   - Prefer QAT/PACT-style training if PTQ loses >0.5 pt. PP-OCR showed QAT was lossless or better (cls 0.9403 → 0.9456, rec 0.6581 → 0.674) ([PP-OCR](https://arxiv.org/pdf/2009.09941)).
   - On iOS, try **weight palettization (6–8-bit)** first. It is the ANE-preferred compression and "you do not lose much accuracy with 6 or 8 bits" ([coremltools](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)).
5. **SE / hardsigmoid parity:** PP-OCR found removing SE from the det backbone gave −36.6% size and −18.8% latency with no HMean loss ([PP-OCR](https://arxiv.org/pdf/2009.09941)). For Paddle-converted weights, the hardsigmoid slope differs between Paddle's SE (0.2) and PyTorch's `hardsigmoid` (1/6) `[UNVERIFIED]` exact values; the repo's recent "configurable SE hardsigmoid slope" commit addresses this. Make sure the exported CoreML/MNN graph uses the same slope, either by exporting as explicit `clip(x*a+b, 0, 1)` or by checking the converter.
6. **Evaluate PP-OCRv6 tiny/small** as a baseline or teacher for parcel labels. It is BN-based, uses re-param DW, a DB det and a CTC rec, and is 1.5M/7.7M params ([PP-OCRv6](https://arxiv.org/html/2606.13108)). Note that it uses **GELU** in the channel mixer, so check ANE/HTP placement. Its internal-set numbers are not comparable with v5, so benchmark on our own data, and on ORT-M4 v6_small was *slower* than v5_mobile.
7. **Drop doc-orientation and unwarping on device** if the capture UI enforces upright framing, and keep a 0.96 MB textline 0/180° classifier only if flipped labels occur in the data ([PaddleOCR docs](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)).
8. **Small text:** do not raise global det resolution. Crop the label ROI first (cheap det pass or the capture-UI guide box), then run det at native resolution inside the ROI. MNN warns that fp16/quant can hurt small-object detection ([MNN](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html)).
9. **Use VLMs offline, not on device:**
   - Run PaddleOCR-VL-1.5/1.6 or HunyuanOCR (outputs text + boxes) on server GPUs to **pseudo-label and hard-mine** parcel-label crops for det/rec training. This is the same recipe PP-OCRv5 used with ERNIE-4.5-VL ([PaddleOCR 3.0](https://arxiv.org/html/2507.05595v1)).
   - Optionally distill det features from a VLM encoder (PP-OCRv5 used GOT's encoder).
   - Gate pseudo-labels by agreement with the classic pipeline, to filter VLM hallucinations.
10. **Fallback path:** for low-confidence or unparseable labels, escalate to a server VLM rather than shipping a ~1.4 GB on-device VLM. The phone VLM evidence shows 22 s/page on a Pixel 8a CPU and degenerate output on S26 CPU ([litert-community](https://huggingface.co/litert-community/PaddleOCR-VL-1.6), `[UNVERIFIED]`). Add deterministic field parsers (tracking numbers, postcodes, check digits) before any model escalation.
11. **ExecuTorch backend choice:**
    - iOS: Core ML delegate with XNNPACK fallback.
    - Android: QNN (Snapdragon) or Vulkan/XNNPACK.
    - MPS no longer appears in the backend list ([ET backends](https://docs.pytorch.org/executorch/stable/backends-overview.html)). Do not build new work on it.
    - Core ML quantization in ET needs iOS 17+, and 4-bit or palettized weights need iOS 18 ([ET-CoreML quant](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-quantization.html)).

---

## Sources

Primary (papers / official docs / official repos):
- Apple ML Research — Deploying Transformers on the Apple Neural Engine: https://machinelearning.apple.com/research/neural-engine-transformers
- Apple ML Research — Deploying Attention-Based Vision Transformers to ANE: https://machinelearning.apple.com/research/vision-transformers
- coremltools — Optimization overview: https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
- Apple WWDC25 session 272 (Vision RecognizeDocumentsRequest): https://developer.apple.com/videos/play/wwdc2025/272/
- ExecuTorch backends overview: https://docs.pytorch.org/executorch/stable/backends-overview.html
- ExecuTorch Core ML troubleshooting: https://docs.pytorch.org/executorch/main/backends/coreml/coreml-troubleshooting.html
- ExecuTorch Core ML quantization: https://docs.pytorch.org/executorch/main/backends/coreml/coreml-quantization.html
- ExecuTorch Qualcomm backend: https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
- ONNX Runtime QNN EP: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Qualcomm QNN HTP op revision history: https://docs.qualcomm.com/bundle/publicresource/80-63442-10/topics/htp_opdef_version_history.md
- LiteRT GPU delegate: https://developers.google.com/edge/litert/performance/gpu
- MNN compression guide: https://mnn-docs.readthedocs.io/en/latest/tools/compress.html
- Google ML Kit Text Recognition v2: https://developers.google.com/ml-kit/vision/text-recognition/v2
- PP-OCR (v1) paper: https://arxiv.org/abs/2009.09941 (PDF tables read: https://arxiv.org/pdf/2009.09941)
- PP-LCNet: https://arxiv.org/html/2109.15099v1
- PP-OCRv3: https://arxiv.org/abs/2206.03001
- PaddleOCR 3.0 technical report: https://arxiv.org/html/2507.05595v1
- PP-OCRv6 technical report: https://arxiv.org/html/2606.13108
- PaddleOCR 3.x OCR pipeline docs: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html
- MobileNetV4: https://arxiv.org/abs/2404.10518
- MobileOne: https://arxiv.org/abs/2206.04040
- RepViT: https://arxiv.org/abs/2307.09283
- FastViT: https://arxiv.org/abs/2303.14189
- ConvNeXt V2: https://arxiv.org/abs/2301.00808 ; repo: https://github.com/facebookresearch/ConvNeXt-V2
- SHViT: https://arxiv.org/abs/2401.16456
- StarNet: https://arxiv.org/abs/2403.19967
- LSNet: https://arxiv.org/abs/2503.23135
- EfficientViT (MIT): https://arxiv.org/abs/2205.14756
- EfficientViT (MSRA): https://arxiv.org/abs/2305.07027
- FAST: https://arxiv.org/abs/2111.02394
- PaddleOCR-VL: https://arxiv.org/abs/2510.14528 ; HTML: https://arxiv.org/html/2510.14528
- PaddleOCR-VL-1.5: https://arxiv.org/abs/2601.21957
- DeepSeek-OCR: https://arxiv.org/abs/2510.18234 ; HTML: https://arxiv.org/html/2510.18234
- DeepSeek-OCR 2: https://arxiv.org/abs/2601.20552
- HunyuanOCR: https://arxiv.org/abs/2511.19575 ; HTML v2: https://arxiv.org/html/2511.19575v2
- MinerU2.5: https://arxiv.org/abs/2509.22186
- MonkeyOCR: https://arxiv.org/abs/2506.05218
- GOT-OCR2.0: https://arxiv.org/abs/2409.01704
- SmolDocling: https://arxiv.org/abs/2503.11576
- granite-docling-258M model card: https://huggingface.co/ibm-granite/granite-docling-258M
- olmOCR: https://arxiv.org/abs/2502.18443
- LightOnOCR: https://arxiv.org/abs/2601.14251 ; blog: https://huggingface.co/blog/lightonai/lightonocr
- GLM-OCR: https://arxiv.org/abs/2603.10910
- Florence-2: https://arxiv.org/abs/2311.06242
- Nougat: https://arxiv.org/abs/2308.13418
- dots.ocr repo: https://github.com/rednote-hilab/dots.ocr
- OmniDocBench repo / leaderboard: https://github.com/opendatalab/OmniDocBench

Secondary (use with care):
- Hugging Face blog — Supercharge your OCR pipelines with open models: https://huggingface.co/blog/ocr-open-models
- litert-community PaddleOCR-VL-1.6 (community conversion, on-device numbers): https://huggingface.co/litert-community/PaddleOCR-VL-1.6
- prasad.tech — Benchmarking on-device OCR (Aug 2026 blog): https://prasad.tech/blog/benchmarking-on-device-ocr/
