"""Benchmark an ONNX model at a fixed shape, optionally via CoreML EP."""

import os
import sys
import argparse
import statistics
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

import numpy as np
import onnxruntime as ort


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="path to .onnx")
    p.add_argument("--dims", required=True, help="e.g. 1,3,1280,704")
    p.add_argument("--ep", default="coreml", choices=["coreml", "cpu"])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                 if args.ep == "coreml" else ["CPUExecutionProvider"])
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, sess_options=sess_opts, providers=providers)
    print(f"providers in use: {sess.get_providers()}")

    input_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(args.seed)
    dims = [int(x) for x in args.dims.split(",")]
    x = rng.standard_normal(dims).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {input_name: x})

    timings_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter_ns()
        sess.run(None, {input_name: x})
        t1 = time.perf_counter_ns()
        timings_ms.append((t1 - t0) / 1e6)

    timings_ms.sort()
    def pct(p):
        k = max(0, min(len(timings_ms) - 1, int(round(p / 100 * (len(timings_ms) - 1)))))
        return timings_ms[k]

    print(
        f"onnx-{args.ep} latency over {args.iters} iters (ms): "
        f"mean={statistics.mean(timings_ms):.2f}  "
        f"p50={pct(50):.2f}  p95={pct(95):.2f}  p99={pct(99):.2f}  "
        f"min={timings_ms[0]:.2f}  max={timings_ms[-1]:.2f}"
    )


if __name__ == "__main__":
    main()
