"""Benchmark an ExecuTorch .pte against the eager PyTorch model.

Measures latency (mean / p50 / p95 / p99 / min / max) over N iterations and
reports max/mean abs diff between eager and .pte outputs for a single fixed
input shape.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

import argparse
import time
import statistics

import torch

from tools.convert_utils import load_config, load_torch_model
from tools.utility import update_rec_head_out_channels
from torchocr.postprocess import build_post_process


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--pte", required=True)
    p.add_argument("--dims", required=True, help="e.g. 1,3,1280,704")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _flatten(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, dict):
        return [v for v in out.values() if isinstance(v, torch.Tensor)]
    if isinstance(out, (list, tuple)):
        flat = []
        for o in out:
            flat.extend(_flatten(o))
        return flat
    return []


def main():
    args = parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    torch.manual_seed(args.seed)
    x = torch.randn(*dims, dtype=torch.float32)

    # ---- eager reference -----------------------------------------------------
    cfg = load_config(args.config)
    post_process_class = build_post_process(cfg["PostProcess"], cfg["Global"])
    update_rec_head_out_channels(cfg, post_process_class)
    eager = load_torch_model(cfg["Architecture"], "cpu", args.weights)
    with torch.no_grad():
        eager_out = eager(x)
    eager_tensors = _flatten(eager_out)
    print(f"eager outputs: {len(eager_tensors)} tensor(s), shapes={[t.shape for t in eager_tensors]}")

    # ---- executorch ----------------------------------------------------------
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(args.pte)
    method = program.load_method("forward")

    # warmup
    for _ in range(args.warmup):
        et_out = method.execute([x])

    et_tensors = _flatten(et_out)
    print(f"ET outputs:    {len(et_tensors)} tensor(s), shapes={[t.shape for t in et_tensors]}")

    # parity
    if len(eager_tensors) == len(et_tensors):
        for i, (a, b) in enumerate(zip(eager_tensors, et_tensors)):
            if a.shape != b.shape:
                print(f"  out[{i}] SHAPE MISMATCH: eager={a.shape} et={b.shape}")
                continue
            diff = (a.float() - b.float()).abs()
            print(f"  out[{i}] max_abs={diff.max().item():.6e}  mean_abs={diff.mean().item():.6e}")
    else:
        print("  parity skipped: output count mismatch")

    # ---- timing --------------------------------------------------------------
    timings_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter_ns()
        method.execute([x])
        t1 = time.perf_counter_ns()
        timings_ms.append((t1 - t0) / 1e6)

    timings_ms.sort()

    def pct(p):
        k = max(0, min(len(timings_ms) - 1, int(round(p / 100 * (len(timings_ms) - 1)))))
        return timings_ms[k]

    print(
        f"\nlatency over {args.iters} iters (ms): "
        f"mean={statistics.mean(timings_ms):.2f}  "
        f"p50={pct(50):.2f}  p95={pct(95):.2f}  p99={pct(99):.2f}  "
        f"min={timings_ms[0]:.2f}  max={timings_ms[-1]:.2f}"
    )


if __name__ == "__main__":
    main()
