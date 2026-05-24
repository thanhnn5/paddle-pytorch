"""Extended detection metrics on a (GT, predictions) pair.

Complementary to visualize_detections.ipynb (single IoU=0.3 P/R/F1). This
script reports:

  1. P/R/F1 swept over IoU thresholds {0.3, 0.5, 0.7, 0.9}
  2. Mean IoU of true-positive matches (localization quality)
  3. Size-stratified P/R/F1 (small/medium/large GT polygons by area)
  4. Per-image FP and FN count histograms (not just ≥1 flags)
  5. Prediction-count drift: |#pred - #gt| distribution

Usage:
  python tools/eval_det_extended.py \\
      --gt   data/ocr_test_20251127/test.txt \\
      --pred output/predictions/sal_mix_balanced_aux/predict_det.txt \\
      --image-dir data/ocr_test_20251127/images
"""
import argparse
import json
import os
import statistics
from collections import Counter
from typing import Dict, List, Tuple

from shapely.geometry import Polygon


# ---------------------------- IO ----------------------------

def load_polygon_file(path: str) -> Dict[str, list]:
    """Read a tab-separated  `image_name<TAB>json_array_of_polygons`  file."""
    out = {}
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            im_name, payload = line.split("\t", 1)
            polys = json.loads(payload)
            out[os.path.basename(im_name)] = polys
    return out


# ---------------------------- Geometry ----------------------------

def _safe_poly(pts):
    p = Polygon(pts)
    return p.buffer(0) if not p.is_valid else p


def polygon_iou(pts_a, pts_b):
    a = _safe_poly(pts_a)
    b = _safe_poly(pts_b)
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def polygon_area(pts):
    return _safe_poly(pts).area


# ---------------------------- Matching ----------------------------

def match_at_iou(gt_pts, pred_pts, iou_thresh: float):
    """Greedy IoU-based matching. Returns (matched_pairs, fn_idx, fp_idx)."""
    pairs = []
    for gi, gp in enumerate(gt_pts):
        for pi, pp in enumerate(pred_pts):
            iou = polygon_iou(gp, pp)
            if iou >= iou_thresh:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    matched_gt, matched_pred = set(), set()
    tp_pairs = []  # list of (gi, pi, iou)
    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        tp_pairs.append((gi, pi, iou))

    fn_idx = [i for i in range(len(gt_pts)) if i not in matched_gt]
    fp_idx = [i for i in range(len(pred_pts)) if i not in matched_pred]
    return tp_pairs, fn_idx, fp_idx


# ---------------------------- Aggregation ----------------------------

def aggregate_at_iou(gt_dict, pred_dict, iou_thresh: float):
    tp = fp = fn = 0
    tp_ious = []
    for im_name, gt_list in gt_dict.items():
        pred_list = pred_dict.get(im_name, [])
        gt_pts = [g["points"] for g in gt_list]
        pred_pts = [p["points"] for p in pred_list]
        tp_pairs, fn_idx, fp_idx = match_at_iou(gt_pts, pred_pts, iou_thresh)
        tp += len(tp_pairs)
        fp += len(fp_idx)
        fn += len(fn_idx)
        tp_ious.extend(iou for _, _, iou in tp_pairs)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return tp, fp, fn, P, R, F, tp_ious


def aggregate_size_stratified(gt_dict, pred_dict, iou_thresh: float):
    """Split GT polygons by area into S/M/L tertiles, report per-bucket recall.

    Also: per-prediction precision split by predicted polygon area.
    """
    # First pass: collect all GT areas to find tertile cutpoints
    gt_areas = []
    for im_name, gt_list in gt_dict.items():
        for g in gt_list:
            gt_areas.append(polygon_area(g["points"]))
    gt_areas.sort()
    n = len(gt_areas)
    cut_s = gt_areas[n // 3] if n else 0
    cut_l = gt_areas[2 * n // 3] if n else 0
    def gt_bucket(a):
        if a < cut_s: return "S"
        if a < cut_l: return "M"
        return "L"

    # Second pass: match and bucket by GT area for recall
    rec_tp = Counter()
    rec_gt_total = Counter()
    for im_name, gt_list in gt_dict.items():
        gt_pts = [g["points"] for g in gt_list]
        pred_pts = [p["points"] for p in pred_dict.get(im_name, [])]
        tp_pairs, fn_idx, _ = match_at_iou(gt_pts, pred_pts, iou_thresh)
        matched_gt_set = {gi for gi, _, _ in tp_pairs}
        for gi, pts in enumerate(gt_pts):
            b = gt_bucket(polygon_area(pts))
            rec_gt_total[b] += 1
            if gi in matched_gt_set:
                rec_tp[b] += 1
    rec = {b: rec_tp[b] / rec_gt_total[b] if rec_gt_total[b] else 0.0
           for b in ("S", "M", "L")}
    return rec, (cut_s, cut_l), rec_gt_total


def per_image_error_histogram(gt_dict, pred_dict, iou_thresh: float):
    fp_counts, fn_counts, count_drift = [], [], []
    for im_name, gt_list in gt_dict.items():
        gt_pts = [g["points"] for g in gt_list]
        pred_pts = [p["points"] for p in pred_dict.get(im_name, [])]
        _, fn_idx, fp_idx = match_at_iou(gt_pts, pred_pts, iou_thresh)
        fp_counts.append(len(fp_idx))
        fn_counts.append(len(fn_idx))
        count_drift.append(len(pred_pts) - len(gt_pts))
    return fp_counts, fn_counts, count_drift


def histogram(values, bins):
    h = Counter()
    for v in values:
        placed = False
        for b in bins:
            if v == b:
                h[str(b)] += 1
                placed = True
                break
        if not placed:
            # Drop into the "+" bucket
            if v > bins[-1]:
                h[f">{bins[-1]}"] += 1
            else:
                h[f"<{bins[0]}"] += 1
    return h


# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT file (tab-separated, json polygons)")
    ap.add_argument("--pred", required=True, help="prediction file (predict_det.txt)")
    ap.add_argument("--image-dir", default=None, help="(unused, for parity with notebook)")
    ap.add_argument("--ious", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9])
    args = ap.parse_args()

    gts = load_polygon_file(args.gt)
    preds = load_polygon_file(args.pred)

    n_images = len(gts)
    n_gt_total = sum(len(v) for v in gts.values())
    n_pred_total = sum(len(v) for v in preds.values())

    print("=" * 72)
    print(f"Images: {n_images}   GT polygons: {n_gt_total}   Pred polygons: {n_pred_total}")
    print("=" * 72)

    # ---- 1. IoU sweep ----
    print("\n[1] P/R/F1 swept over IoU threshold")
    print(f"  {'IoU':>5}  {'TP':>5} {'FP':>5} {'FN':>5}  "
          f"{'P':>6}  {'R':>6}  {'F1':>6}  {'meanIoU(TP)':>11}")
    by_iou = {}
    for thr in args.ious:
        tp, fp, fn, P, R, F, ious = aggregate_at_iou(gts, preds, thr)
        m = sum(ious) / len(ious) if ious else 0.0
        by_iou[thr] = (tp, fp, fn, P, R, F, m, ious)
        print(f"  {thr:>5.2f}  {tp:>5} {fp:>5} {fn:>5}  "
              f"{P:>6.4f}  {R:>6.4f}  {F:>6.4f}  {m:>11.4f}")

    # ---- 2. Localization-quality detail (at IoU 0.3, the matching threshold) ----
    _, _, _, _, _, _, _, ious_at_03 = by_iou[args.ious[0]]
    if ious_at_03:
        ious_at_03 = sorted(ious_at_03)
        q = lambda p: ious_at_03[int(p * (len(ious_at_03) - 1))]
        print("\n[2] TP-IoU distribution (matches @ IoU>={:.2f})".format(args.ious[0]))
        print(f"  n={len(ious_at_03)}  mean={sum(ious_at_03)/len(ious_at_03):.4f}")
        print(f"  p10={q(0.10):.4f}  p25={q(0.25):.4f}  p50={q(0.50):.4f}  "
              f"p75={q(0.75):.4f}  p90={q(0.90):.4f}")

    # ---- 3. Size-stratified recall ----
    print("\n[3] Recall stratified by GT polygon area (S/M/L tertiles, IoU>=0.5)")
    rec, (cs, cl), totals = aggregate_size_stratified(gts, preds, 0.5)
    print(f"  cut points (px²):  S<{cs:.0f}   M:[{cs:.0f},{cl:.0f})   L>={cl:.0f}")
    for b in ("S", "M", "L"):
        print(f"  {b}: recall={rec[b]:.4f}  (n_gt={totals[b]})")

    # ---- 4. Per-image FP/FN histograms ----
    fp_counts, fn_counts, drift = per_image_error_histogram(gts, preds, 0.5)
    print("\n[4] Per-image error counts (IoU>=0.5)")
    fp_hist = histogram(fp_counts, [0, 1, 2, 3, 4, 5])
    fn_hist = histogram(fn_counts, [0, 1, 2, 3, 4, 5])
    print("  FP per image:")
    for k in ["0", "1", "2", "3", "4", "5", ">5"]:
        c = fp_hist.get(k, 0)
        print(f"    {k:>3}: {c:>5}  ({100*c/n_images:>5.1f}%)")
    print("  FN per image:")
    for k in ["0", "1", "2", "3", "4", "5", ">5"]:
        c = fn_hist.get(k, 0)
        print(f"    {k:>3}: {c:>5}  ({100*c/n_images:>5.1f}%)")

    # ---- 5. Prediction count drift ----
    print("\n[5] Prediction-count drift  (#pred - #gt per image)")
    print(f"  mean={statistics.mean(drift):+.3f}   "
          f"median={statistics.median(drift):+}   "
          f"stdev={statistics.stdev(drift) if len(drift) > 1 else 0:.3f}")
    drift_hist = histogram(drift, list(range(-3, 4)))
    print("  Drift distribution:")
    for k in ["<-3", "-3", "-2", "-1", "0", "1", "2", "3", ">3"]:
        c = drift_hist.get(k, 0)
        print(f"    {k:>4}: {c:>5}  ({100*c/n_images:>5.1f}%)")


if __name__ == "__main__":
    main()
