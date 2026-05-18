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
    p.add_argument("--group-seeds", action="store_true",
                   help="aggregate runs that share all hparams except --seed; "
                        "show mean ± std across seeds")
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

    if args.group_seeds:
        rows = _group_by_hparams(rows)

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


def _group_by_hparams(rows):
    """Collapse rows sharing the same hparams (except name/seed) into one row
    with mean values for the numeric metrics. `name` becomes `<base>×N` where
    N is the seed count."""
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    hparam_keys = ("lr", "img", "bs", "cos_w", "stages", "size")
    metric_keys = ("cos_first", "cos_last", "mse_ln_first", "mse_ln_last",
                   "steps", "wall_s")
    for r in rows:
        key = tuple(r[k] for k in hparam_keys)
        groups[key].append(r)

    out = []
    for key, items in groups.items():
        merged = {k: items[0][k] for k in hparam_keys}
        merged["name"] = f"lr{key[0]}_img{key[1]}_cw{key[3]}_st{key[4]} ×{len(items)}"
        for mk in metric_keys:
            vals = [it[mk] for it in items]
            merged[mk] = statistics.fmean(vals)
            if len(vals) > 1:
                merged[mk + "_std"] = statistics.stdev(vals)
        out.append(merged)
    return out


if __name__ == "__main__":
    main()
