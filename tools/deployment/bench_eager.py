"""Benchmark PyTorch eager forward (CPU or MPS) at a fixed shape."""

import os
import sys
import argparse
import statistics
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..")))

import torch

from torchocr import Config
from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils.ckpt import load_ckpt
from tools.utility import update_rec_head_out_channels, ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--dims", required=True, help="e.g. 1,3,1280,704")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    args = parse_args()
    cfg = Config(args.config)
    cfg.merge_dict(args.opt)
    _cfg = cfg.cfg

    post_process_class = build_post_process(_cfg["PostProcess"])
    update_rec_head_out_channels(_cfg, post_process_class)
    model = build_model(_cfg["Architecture"])
    load_ckpt(model, _cfg)
    model.eval()

    device = args.device
    model = model.to(device)
    torch.manual_seed(args.seed)
    dims = [int(x) for x in args.dims.split(",")]
    x = torch.randn(*dims, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
            _sync(device)

        timings_ms = []
        for _ in range(args.iters):
            t0 = time.perf_counter_ns()
            _ = model(x)
            _sync(device)
            t1 = time.perf_counter_ns()
            timings_ms.append((t1 - t0) / 1e6)

    timings_ms.sort()
    def pct(p):
        k = max(0, min(len(timings_ms) - 1, int(round(p / 100 * (len(timings_ms) - 1)))))
        return timings_ms[k]

    print(
        f"eager-{device} latency over {args.iters} iters (ms): "
        f"mean={statistics.mean(timings_ms):.2f}  "
        f"p50={pct(50):.2f}  p95={pct(95):.2f}  p99={pct(99):.2f}  "
        f"min={timings_ms[0]:.2f}  max={timings_ms[-1]:.2f}"
    )


if __name__ == "__main__":
    main()
