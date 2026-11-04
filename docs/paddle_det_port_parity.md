# PaddleOCR → torch → ExecuTorch: detection parity findings

What we verified when porting the Paddle-trained address-numeral detectors to
torchocr and ExecuTorch (CoreML), and what to watch for when doing it again.

| Model | Paddle source | torch weights | ExecuTorch |
|---|---|---|---|
| PP-OCRv6 small det | `gs://data-management-data-ai-models/models/address-numeral/detection/v6_small_mask-3-1` | `PP-OCRv6_small-det_mask-3-1/best_accuracy.pth` | `PP-OCRv6_small-det_mask-3-1/executorch/model_coreml.pte` (fp16) |
| PP-OCRv5 mobile det | `PP-OCRv5_mobile-det_mask-3_exp2/inference.*` | `PP-OCRv5_mobile-det_mask-3_exp2/mob_det_2.pth` | `.../executorch/model_coreml.pte` (fp16), `.../executorch_fp32/model_coreml.pte` (fp32) |

All `.pte` files take a static `[1, 3, 1280, 704]` input and target iOS 17.

## Summary

- **torch matches Paddle.** With the fixes below, both models give the same
  boxes as Paddle inference on all 19 images in `doc/imgs`, with 0 px shift.
- **SE gate slope must be 0.2 for Paddle weights.** PaddleOCR's det `SEModule`
  computes `hardsigmoid(slope=0.2)`. torchocr used `nn.Hardsigmoid` (slope 1/6),
  which broke the v5 port: max probability error 0.997, and 17 of 94 boxes lost
  or changed. Set `Neck.hardsigmoid_slope: 0.2`; the PP-OCRv5 mobile config now
  does.
- **EMA has no effect at export time.** With `use_ema: true`, Paddle saves the
  EMA weights into `.pdparams`, so the converted `.pth` and the `.pte` already
  carry them. The torch `ModelEMA` matches Paddle's to within 7e-7.
- **fp16 CoreML costs boxes; fp32 doesn't.** fp32 `.pte` files match Paddle
  exactly. fp16 loses 7/94 boxes on v5 and 1/48 on v6. Use
  `export_executorch.py --precision fp32` when parity matters, and measure
  latency on a device first.
- **The old `mob_det_2_coreml.pte` is wrong.** It finds 0 of Paddle's 94 boxes
  and agrees on only 90.5% of mask pixels. Replace it with one of the new
  exports.

## 1. Weight conversion (v6)

`tools/deployment/convert_paddle_weights.py` converts `.pdparams` to `.pth` and
checks the result against the Paddle inference model:

```bash
python tools/deployment/convert_paddle_weights.py \
    -c configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    --paddle_params PP-OCRv6_small-det_mask-3-1/best_accuracy.pdparams \
    --output PP-OCRv6_small-det_mask-3-1/best_accuracy.pth \
    --paddle_infer_dir PP-OCRv6_small-det_mask-3-1 \
    --image doc/imgs/1.jpg --shape 1280,704
```

- **The mapping is a pure rename.** `._mean` → `.running_mean` and
  `._variance` → `.running_var`. All 514 Paddle tensors map 1:1, and the
  conversion fails on any missing key, extra key or shape mismatch.
- **Max diff vs Paddle is 5.7e-3; mean diff is 1e-6.** The max comes from
  Paddle's CPU kernels, not the port:
  - The inference graph's ops and attributes match the port: hardsigmoid
    slopes, exact GELU, SAME-padded max-pool, nearest upsampling.
  - The weights in `inference.pdiparams` are bit-identical to
    `best_accuracy.pdparams`.
  - Paddle's pre-fused weights match an fp64 re-fusion to about 1e-7 relative.
  - torch fp32 is within 2e-5 of an fp64 run. Paddle is 5.7e-3 away from it,
    with IR optimisation on or off.
- **`rep()` is lossless.** The re-parameterised model used for export is within
  6e-6 of the unfused one.

## 2. Hardsigmoid slope

PaddleOCR has two hardsigmoid slopes, and the port must use the same one in
each place. Op counts in Paddle's `inference.json`:

| Model | slope 1/6 (backbone SE: PPLCNetV3/V4) | slope 0.2 (neck SE: RSEFPN / RepLKFPN) |
|---|---|---|
| v5 mobile | 2 | 8 |
| v6 small | 5 | 8 |

**Where the mismatch was.** Paddle's `RSEFPN` reuses
`det_mobilenet_v3.SEModule`, which calls
`F.hardsigmoid(x, slope=0.2, offset=0.5)`. torchocr copied that import, but its
`SEModule` uses `nn.Hardsigmoid`, which is slope 1/6.

**Effect on outputs.** Same weights in each pair; only the slope of the 8 neck
SE blocks differs. Measured against Paddle inference on 19 images at 1280×704,
with `PostProcess` from each model's `inference.yml`:

| Model | Neck SE slope | Max prob diff | Boxes (Paddle / torch) | Unmatched | Max shift |
|---|---|---|---|---|---|
| v5 | 1/6 (old torchocr) | 0.997 | 94 / 86 | 17 | 9 px |
| v5 | **0.2** | 3.2e-4 | 94 / 94 | 0 | 0 px |
| v6 | 1/6 | 0.675 | 48 / 47 | 1 | 4 px |
| v6 | **0.2** | 5.7e-3 | 48 / 48 | 0 | 0 px |

"Unmatched" means the Paddle box has no torch box within 10 px.

**The fix.**
- `SEModule(hardsigmoid_slope=...)` and `RSEFPN(hardsigmoid_slope=...)`. The
  default, `None`, keeps `nn.Hardsigmoid`. Models trained in torchocr learned
  with slope 1/6 and must keep it, which is why the default didn't change. The
  ConvNeXt/femto configs that use `RSEFPN` are torch-trained and leave it
  unset.
- `configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml` sets
  `Neck.hardsigmoid_slope: 0.2`.
- `RepLKFPN` (v6) always uses 0.2.
- The slope has no parameters, so existing checkpoints load either way.
- The MobileNetV3 det backbone uses the same `SEModule`, so for Paddle weights
  it is also off by default. No config of ours uses it with Paddle weights, so
  the option isn't wired into that backbone.
- Tests: `tests/test_se_hardsigmoid_slope.py`.

## 3. EMA

The v6 Paddle config trains with `use_ema: true`, `ema_decay: 0.9997` and
`ema_decay_type: threshold`.

- **What's in the checkpoint.** Before saving, `tools/program.py` swaps the EMA
  weights in (`model.set_state_dict(ema.apply())`). So `best_accuracy.pdparams`,
  and `inference.pdiparams` exported from it, hold the EMA weights. The raw
  training weights go to a separate `.pdema` file.
- **No `.pdema` on GCS.** Without the raw weights, we can't measure how much EMA
  improved this checkpoint. Conversion and export are unaffected.
- **torch `ModelEMA` (`torchocr/utils/ema.py`) matches Paddle's.** The same
  3,000-step trajectory of conv and BN weights, including running stats, went
  through both implementations. The averaged weights agreed to within 7e-7 at
  steps 1, 5, 50, 500 and 3000. They differed from the raw weights by up to
  0.21, so EMA does change what gets saved.
- **The torch trainer matches Paddle's behaviour.** It updates EMA after every
  optimizer step, evaluates and saves the EMA weights, and stores the raw
  weights plus EMA state so a resume is exact.

## 4. ExecuTorch / CoreML export

```bash
python tools/deployment/export_executorch.py -c <config> --backend coreml \
    [--precision fp16|fp32] [--ios 17] \
    -o Global.pretrained_model=<.pth> Export.export_dir=<dir> \
       "Export.export_shape=[1,3,1280,704]"
```

Measured against Paddle inference on 19 images at 1280×704. Latency is the
median on the dev MacBook, not a phone:

| Model | Runtime | Max prob diff | Boxes (Paddle / model) | Unmatched | Max shift | Latency |
|---|---|---|---|---|---|---|
| v5 | torch eager | 3.2e-4 | 94 / 94 | 0 | 0 px | 174 ms |
| v5 | CoreML fp16 | 0.721 | 94 / 95 | 7 | 5 px | 6.5 ms |
| v5 | CoreML fp32 | 2.8e-4 | 94 / 94 | 0 | 0 px | 6.8 ms |
| v5 | old `mob_det_2_coreml.pte` | 1.0 | 94 / 0 | 94 | n/a | n/a |
| v6 | torch eager | 5.7e-3 | 48 / 48 | 0 | 0 px | 152 ms |
| v6 | CoreML fp16 | 0.185 | 48 / 47 | 1 | 3 px | 8.4 ms |
| v6 | CoreML fp32 | 5.7e-3 | 48 / 48 | 0 | 0 px | 10.5 ms |

**Reading the table.**
- The whole network is delegated to CoreML: the graph's only non-delegated node
  is its output.
- The fp16 loss is just rounding. The fp32 build of the same graph is exact.
- v5 is more sensitive to fp16 than v6. Its lost boxes are ones Paddle scores
  near `box_thresh`. For example, Paddle scores one v6 box on `00009282.jpg` at
  0.4519 and fp16 at 0.443, against `box_thresh` 0.45.
- fp16 is the default because CoreML can run it on the Neural Engine, while
  fp32 runs on GPU or CPU. On a Mac both take about the same time.
- **Before choosing fp32, measure on the target iPhone.**

## 5. Gotchas

- **Never import Paddle and torch in the same process (macOS).** Depending on
  import order it hangs or aborts. `convert_paddle_weights.py` runs Paddle in a
  child process and passes results back through `.npz`.
- **Import order also matters for coremltools.** Importing `coremltools` before
  torch/torchocr segfaults. Import `export_executorch` (or torch) first.
- **`export_executorch.export_single_model` reads a module-level `cfg`.** That
  variable is only set when the script runs as `__main__`. If you call it from
  Python, set `export_executorch.cfg` first.
- **Static input shape.** `tools/infer/executorch/predict_det.py` resizes by
  aspect ratio, so a static `.pte` only accepts images that resize to exactly
  1280×704 (9:16 photos do). Anything else fails with `set_inputs ... 0x10`.
- **Padding in the v6 stem.** Paddle's `padding="SAME"` on the stride-1 2×2
  convs and the max-pool pads asymmetrically, (0, 1). The port uses an explicit
  `F.pad` instead of torch's `padding="same"`, so the exported graph only
  contains a plain conv and pool.
- **`transformers` is now imported lazily.** The ConvNeXt backbones load it
  only when built, so a broken `transformers`/`tokenizers` install no longer
  breaks every det model.
