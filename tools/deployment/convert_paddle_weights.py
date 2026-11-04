"""Convert a PaddleOCR training checkpoint (.pdparams) to a torchocr .pth and
verify numerical parity against the Paddle inference model.

Works for models whose torchocr port mirrors Paddle parameter names (e.g.
PP-OCRv6 det: PPLCNetV4 + RepLKFPN + DBHead). Mapping is a pure rename:
``._mean`` -> ``.running_mean``, ``._variance`` -> ``.running_var``. Conversion
is strict: any missing / unexpected key or shape mismatch aborts.

Example:
    python tools/deployment/convert_paddle_weights.py \
        -c configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
        --paddle_params PP-OCRv6_small-det_mask-3-1/best_accuracy.pdparams \
        --output PP-OCRv6_small-det_mask-3-1/best_accuracy.pth \
        --paddle_infer_dir PP-OCRv6_small-det_mask-3-1 \
        --image doc/imgs/1.jpg --shape 1280,704

Verification (``--paddle_infer_dir``) runs, on the same input:
  1. Paddle inference (inference.json + inference.pdiparams)  <- reference
  2. torch eager, un-reparameterised
  3. torch eager after ``rep()`` (what export_executorch.py exports)
and reports max/mean abs diff of the probability map.

Expected numbers (PP-OCRv6 small det, 1280x704): mean|d| ~1e-6, max|d| ~6e-3
on a handful of p~0.5 edge pixels, identical post-processed boxes. The max
diff is Paddle's CPU-kernel accumulation, not the port: torch fp32 matches an
fp64 run of the same weights to 2e-5, and Paddle's fused inference weights
match an fp64 re-fusion of the .pdparams to ~1e-7 relative.

Paddle and torch/torchocr are never imported in the same process: on macOS
loading both deadlocks or aborts depending on import order. All Paddle work
runs in a child process (``--_paddle_worker``) that hands results back via .npz.
"""
import argparse
import os
import subprocess
import sys
import tempfile

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Paddle side (child process only; never imports torch)
# --------------------------------------------------------------------------
def paddle_worker(params_path, infer_dir, inputs_npz, out_npz):
    import paddle

    sd = paddle.load(params_path)
    result = {f"param::{k}": (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
              for k, v in sd.items()}

    if infer_dir:
        import paddle.inference as pdi

        model_file = os.path.join(infer_dir, "inference.json")
        if not os.path.exists(model_file):
            model_file = os.path.join(infer_dir, "inference.pdmodel")
        cfg = pdi.Config(model_file, os.path.join(infer_dir, "inference.pdiparams"))
        cfg.disable_gpu()
        cfg.switch_ir_optim(False)  # no fusion passes: closest to the raw graph
        if hasattr(cfg, "disable_mkldnn"):
            cfg.disable_mkldnn()
        pred = pdi.create_predictor(cfg)
        inp = pred.get_input_handle(pred.get_input_names()[0])
        out = pred.get_output_handle(pred.get_output_names()[0])
        for name, x in np.load(inputs_npz).items():
            inp.reshape(x.shape)
            inp.copy_from_cpu(np.ascontiguousarray(x))
            pred.run()
            result[f"ref::{name}"] = out.copy_to_cpu()
    np.savez(out_npz, **result)


def run_paddle(params_path, infer_dir, inputs):
    with tempfile.TemporaryDirectory() as tmp:
        in_npz, out_npz = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(in_npz, **inputs)
        cmd = [sys.executable, os.path.abspath(__file__), "--_paddle_worker",
               "--paddle_params", params_path, "--_inputs", in_npz, "--_out", out_npz]
        if infer_dir:
            cmd += ["--paddle_infer_dir", infer_dir]
        subprocess.run(cmd, check=True)
        data = dict(np.load(out_npz))
    params = {k[len("param::"):]: v for k, v in data.items() if k.startswith("param::")}
    refs = {k[len("ref::"):]: v for k, v in data.items() if k.startswith("ref::")}
    return params, refs


# --------------------------------------------------------------------------
# Torch side
# --------------------------------------------------------------------------
def paddle_to_torch_name(name):
    return name.replace("._mean", ".running_mean").replace("._variance", ".running_var")


def convert(model, pd_params):
    import torch

    th_sd = model.state_dict()
    new_sd, unexpected, mismatched = {}, [], []
    for k, arr in pd_params.items():
        tk = paddle_to_torch_name(k)
        if tk not in th_sd:
            unexpected.append(k)
            continue
        if tuple(arr.shape) != tuple(th_sd[tk].shape):
            mismatched.append((k, tuple(arr.shape), tuple(th_sd[tk].shape)))
            continue
        new_sd[tk] = torch.from_numpy(np.ascontiguousarray(arr)).to(th_sd[tk].dtype)
    missing = [k for k in th_sd if k not in new_sd and not k.endswith("num_batches_tracked")]
    if unexpected or mismatched or missing:
        raise RuntimeError(
            f"Conversion not 1:1\n  unexpected paddle keys: {unexpected}\n"
            f"  shape mismatches: {mismatched}\n  missing torch keys: {missing}")
    for k in th_sd:  # keep torch-only buffers (num_batches_tracked)
        new_sd.setdefault(k, th_sd[k])
    model.load_state_dict(new_sd, strict=True)
    print(f"converted {len(pd_params)} paddle tensors -> {len(new_sd)} torch entries (strict, 1:1)")
    return new_sd


def make_input(image, h, w):
    if image:
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(image)
        img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0  # BGR, like DecodeImage(BGR)
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        img = (img - mean) / std
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])
    return np.random.RandomState(0).randn(1, 3, h, w).astype(np.float32)


def torch_infer(model, x):
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(x))["res"].numpy()


def report(name, ref, out, prob_thresh):
    d = np.abs(ref - out)
    agree = ((ref > prob_thresh) == (out > prob_thresh)).mean()
    print(f"  {name:<24s} max|d|={d.max():.3e}  mean|d|={d.mean():.3e}  "
          f"mask agreement @>{prob_thresh}: {agree * 100:.4f}%")
    return d.max()


def main(args):
    h, w = map(int, args.shape.split(","))
    inputs = {}
    if args.paddle_infer_dir:
        if args.image:
            inputs["image"] = make_input(args.image, h, w)
        inputs["random"] = make_input(None, h, w)
    pd_params, refs = run_paddle(args.paddle_params, args.paddle_infer_dir, inputs)

    import copy
    import torch
    from torchocr import Config
    from torchocr.modeling.architectures import build_model

    cfg = Config(args.config).cfg
    model = build_model(copy.deepcopy(cfg["Architecture"]))
    model.eval()
    sd = convert(model, pd_params)
    out_path = args.output or os.path.splitext(args.paddle_params)[0] + ".pth"
    torch.save({"state_dict": sd}, out_path)
    print(f"saved {out_path}")
    if not refs:
        return 0

    repped = copy.deepcopy(model)
    for m in list(repped.modules()):
        if hasattr(m, "rep") and not getattr(m, "is_repped", True):
            m.rep()

    thresh = cfg.get("PostProcess", {}).get("thresh", 0.3)
    ok = True
    for name, x in inputs.items():
        ref = refs[name]
        print(f"[{name} input {list(x.shape)}] paddle out {list(ref.shape)}, "
              f"prob range [{ref.min():.4f}, {ref.max():.4f}]")
        eager = torch_infer(model, x)
        out_rep = torch_infer(repped, x)
        ok &= report("torch eager vs paddle", ref, eager, thresh) <= args.atol
        ok &= report("torch rep() vs paddle", ref, out_rep, thresh) <= args.atol
        report("torch rep() vs eager", eager, out_rep, thresh)
    print("PARITY OK" if ok else f"PARITY FAILED (atol={args.atol})")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config")
    ap.add_argument("--paddle_params", required=True)
    ap.add_argument("--output", default=None, help="default: <paddle_params>.pth")
    ap.add_argument("--paddle_infer_dir", default=None,
                    help="dir with inference.json/.pdiparams to verify against")
    ap.add_argument("--image", default=None, help="verification image (a random input is always added)")
    ap.add_argument("--shape", default="1280,704", help="verification H,W (multiples of 32)")
    ap.add_argument("--atol", type=float, default=1e-2,
                    help="max abs diff allowed on the prob map (see module docstring)")
    ap.add_argument("--_paddle_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_inputs", help=argparse.SUPPRESS)
    ap.add_argument("--_out", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._paddle_worker:
        paddle_worker(a.paddle_params, a.paddle_infer_dir, a._inputs, a._out)
    else:
        if not a.config:
            ap.error("-c/--config is required")
        sys.exit(main(a))
