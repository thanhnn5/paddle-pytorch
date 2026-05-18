"""Aggregate summary.json files from sweep runs into a comparison table.

Usage:
    python tools/distill/compare.py output/sweep_*/summary.json
"""

import argparse
import glob
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+",
                   help="summary.json paths or glob (e.g. 'output/sweep_*/summary.json')")
    p.add_argument("--sort-by", default="cos_last",
                   help="cos_last | cos_first | mse_ln_last | mse_ln_first | steps")
    p.add_argument("--sort-stage", type=int, default=-1,
                   help="which align-stage index to sort by (-1 = last)")
    args = p.parse_args()

    paths = []
    for pat in args.paths:
        if any(c in pat for c in "*?["):
            paths.extend(glob.glob(pat))
        else:
            paths.append(pat)
    paths = sorted(set(paths))
    if not paths:
        print("no summary.json files matched", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in paths:
        with open(path) as f:
            s = json.load(f)
        a = s["args"]
        fp = s["final_probe"]
        idx = args.sort_stage
        rows.append({
            "name": os.path.basename(os.path.dirname(path)),
            "lr": a["lr"],
            "img": a["img_size"],
            "bs": a["batch_size"],
            "cos_w": a.get("cosine_weight", 0.0),
            "stages": "+".join(str(s) for s in a["align_stages"]),
            "size": a.get("student_size", "?"),
            "steps": s["total_steps"],
            "wall_s": s["wallclock_s"],
            "cos_first": fp["cos"][0],
            "cos_last": fp["cos"][idx],
            "mse_ln_first": fp["mse_ln"][0],
            "mse_ln_last": fp["mse_ln"][idx],
        })

    key = args.sort_by
    reverse = key.startswith("cos")           # higher = better for cosine
    rows.sort(key=lambda r: r[key], reverse=reverse)

    cols = [
        ("name", 30),
        ("size", 6),
        ("lr", 8),
        ("img", 5),
        ("bs", 4),
        ("cos_w", 6),
        ("stages", 8),
        ("steps", 7),
        ("wall_s", 7),
        ("cos_first", 9),
        ("cos_last", 9),
        ("mse_ln_first", 12),
        ("mse_ln_last", 12),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(_fmt(r[name], w) for name, w in cols)
        print(line)


def _fmt(v, w):
    if isinstance(v, float):
        if abs(v) < 1e-2 or abs(v) >= 1e3:
            s = f"{v:.2e}"
        else:
            s = f"{v:.4f}"
    else:
        s = str(v)
    return s.ljust(w)[:w]


if __name__ == "__main__":
    main()
