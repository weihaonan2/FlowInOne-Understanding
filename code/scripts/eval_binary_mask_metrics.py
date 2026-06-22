#!/usr/bin/env python3
"""Standard binary mask evaluation for RefCOCOg / ReasonSeg / SA-CO.

Computes:
  cIoU       — cumulative IoU  (Σ intersection / Σ union)
  gIoU       — mean per-sample IoU
  mDice      — mean per-sample Dice
  P@0.5      — fraction of samples with IoU >= 0.5
  P@0.7      — fraction of samples with IoU >= 0.7
  per-sample — individual IoU, Dice, pred_area, gt_area, etc.

Inputs:
  --pred-dir           directory of prediction PNGs (each file: a sample)
  --gt-wds             WDS tar glob/brace pattern for GT masks
  --gt-dir             directory of GT PNGs (alternative to --gt-wds)
  --manifest           optional JSON manifest to match pred↔gt keys
  --threshold          binarization threshold (default 0.5)
  --sweep-thresholds   comma-separated thresholds, e.g. 0.1,0.2,0.3
  --no-invert          disable auto-inversion detection
  --postprocess        apply largest-connected-component postprocessing

Outputs (written to --output-dir):
  metrics.json       — full structured results
  summary.csv        — one-line summary
  per_sample.csv     — per-sample metrics
  threshold_sweep.csv — if --sweep-thresholds
  visuals/           — if --save-visuals, side-by-side PNGs
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard binary mask evaluation")
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory of prediction PNGs")
    parser.add_argument("--gt-wds", default=None,
                        help="Glob/brace pattern for GT WDS tar shards")
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Directory of GT PNGs (alternative to --gt-wds)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="JSON manifest for key matching")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for metrics files")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold [0-1]")
    parser.add_argument("--sweep-thresholds", default=None,
                        help="Comma-separated thresholds, e.g. 0.1,0.2,...,0.9")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply largest-component postprocessing")
    parser.add_argument("--save-visuals", action="store_true",
                        help="Save side-by-side visual comparison PNGs")
    parser.add_argument("--max-visuals", type=int, default=200,
                        help="Max number of visualizations")
    parser.add_argument("--name", default="eval", help="Experiment name for reports")
    return parser.parse_args()


# ── loading ────────────────────────────────────────────────────────────────
def expand(pattern: str) -> list[Path]:
    try:
        import braceexpand
        patterns = list(braceexpand.braceexpand(pattern))
    except Exception:
        patterns = [pattern]
    paths: list[Path] = []
    for item in patterns:
        paths.extend(Path(p) for p in sorted(glob.glob(item)))
    return sorted(dict.fromkeys(paths))


def image_to_mask(image: Image.Image, threshold: float) -> np.ndarray:
    """Convert RGB/grayscale PNG to binary mask."""
    arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return arr >= threshold


def load_gt_from_wds(pattern: str, threshold: float) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for tar_path in expand(pattern):
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if name.endswith(".out.png") or name.endswith(".out.jpg"):
                    key = Path(name.rsplit(".", 2)[0]).name
                else:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                masks[key] = image_to_mask(Image.open(io.BytesIO(f.read())), threshold)
    return masks


def load_gt_from_dir(gt_dir: Path, threshold: float) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for path in gt_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            masks[path.stem] = image_to_mask(Image.open(path), threshold)
    return masks


def collect_predictions(pred_dir: Path) -> dict[str, Path]:
    preds: dict[str, Path] = {}
    for path in pred_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            preds[path.stem] = path
    return preds


# ── postprocessing ─────────────────────────────────────────────────────────
def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (4-connectivity)."""
    try:
        from scipy import ndimage
        labeled, num = ndimage.label(mask)
        if num == 0:
            return mask
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        largest_label = np.argmax(sizes) + 1
        return labeled == largest_label
    except ImportError:
        # Fallback: simple connected component via iterative fill
        return mask  # no-op if scipy unavailable


# ── metrics ────────────────────────────────────────────────────────────────
def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute per-sample metrics.  Resizes pred→gt if shapes differ."""
    if pred.shape != gt.shape:
        pred_img = Image.fromarray((pred.astype(np.uint8) * 255), mode="L")
        pred_img = pred_img.resize(gt.shape[::-1], Image.NEAREST)
        pred = np.asarray(pred_img) > 127

    intersection = np.logical_and(pred, gt).sum(dtype=np.float64)
    pred_area = pred.sum(dtype=np.float64)
    gt_area = gt.sum(dtype=np.float64)
    union = pred_area + gt_area - intersection

    iou = float(intersection / union) if union > 0 else (1.0 if pred_area == 0 and gt_area == 0 else 0.0)
    dice = float(2.0 * intersection / (pred_area + gt_area)) if (pred_area + gt_area) > 0 else (1.0 if pred_area == 0 and gt_area == 0 else 0.0)
    precision = float(intersection / pred_area) if pred_area > 0 else (1.0 if gt_area == 0 else 0.0)
    recall = float(intersection / gt_area) if gt_area > 0 else (1.0 if pred_area == 0 else 0.0)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "intersection": float(intersection),
        "union": float(union),
        "pred_area": float(pred_area),
        "gt_area": float(gt_area),
    }


def aggregate_metrics(per_sample: list[dict]) -> dict:
    """Compute cIoU, gIoU, mDice, P@0.5, P@0.7 from per-sample list."""
    n = len(per_sample)
    if n == 0:
        return {"cIoU": 0.0, "gIoU": 0.0, "mDice": 0.0, "P@0.5": 0.0, "P@0.7": 0.0,
                "num_samples": 0}

    sum_intersection = sum(s["intersection"] for s in per_sample)
    sum_union = sum(s["union"] for s in per_sample)
    ciou = float(sum_intersection / sum_union) if sum_union > 0 else 0.0
    giou = float(np.mean([s["iou"] for s in per_sample]))
    mdice = float(np.mean([s["dice"] for s in per_sample]))
    p05 = float(np.mean([1.0 if s["iou"] >= 0.5 else 0.0 for s in per_sample]))
    p07 = float(np.mean([1.0 if s["iou"] >= 0.7 else 0.0 for s in per_sample]))

    return {
        "cIoU": ciou,
        "gIoU": giou,
        "mDice": mdice,
        "P@0.5": p05,
        "P@0.7": p07,
        "num_samples": n,
    }


# ── visuals ────────────────────────────────────────────────────────────────
def make_visual(pred: np.ndarray, gt: np.ndarray, key: str) -> Image.Image:
    """Create a side-by-side comparison: GT | Prediction | Overlay."""
    h, w = gt.shape
    # Ensure same shape
    if pred.shape != (h, w):
        pred_img = Image.fromarray((pred.astype(np.uint8) * 255), mode="L").resize((w, h), Image.NEAREST)
        pred = np.asarray(pred_img) > 127

    # Create RGB tiles
    def to_rgb(mask, color):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[mask] = color
        return rgb

    gt_rgb = to_rgb(gt, (0, 255, 0))        # green
    pred_rgb = to_rgb(pred, (255, 0, 0))     # red

    # Overlay: green=GT, red=pred, yellow=overlap
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, np.logical_not(gt))
    fn = np.logical_and(np.logical_not(pred), gt)
    overlay[tp] = (255, 255, 0)   # yellow
    overlay[fp] = (255, 0, 0)     # red
    overlay[fn] = (0, 255, 0)     # green

    # Concatenate horizontally
    panel = np.concatenate([gt_rgb, pred_rgb, overlay], axis=1)
    img = Image.fromarray(panel)

    # Add labels
    draw = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, label in enumerate(["GT", "Pred", "Overlay"]):
        x = i * w + 4
        draw.text((x, 2), f"{label} | {key}", fill=(255, 255, 255), font=font)

    return img


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if (args.gt_wds is None) == (args.gt_dir is None):
        raise SystemExit("pass exactly one of --gt-wds or --gt-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load GT
    print(f"Loading GT masks...", flush=True)
    gt_masks = (
        load_gt_from_wds(args.gt_wds, 0.5)
        if args.gt_wds
        else load_gt_from_dir(args.gt_dir, 0.5)
    )
    print(f"  {len(gt_masks)} GT masks loaded", flush=True)

    # Collect predictions
    pred_paths = collect_predictions(args.pred_dir)
    print(f"  {len(pred_paths)} predictions found", flush=True)

    if not pred_paths:
        raise SystemExit(f"No predictions found in {args.pred_dir}")

    def evaluate_at_threshold(thresh: float) -> tuple[dict, list[dict], list[str]]:
        per_sample = []
        missing = []
        for key, gt in sorted(gt_masks.items()):
            pp = pred_paths.get(key)
            if pp is None:
                missing.append(key)
                continue
            pred = image_to_mask(Image.open(pp), thresh)

            if args.postprocess:
                pred = largest_component(pred)

            row = {"sample_id": key, "pred_path": str(pp)}
            row.update(compute_sample_metrics(pred, gt))
            per_sample.append(row)

        agg = aggregate_metrics(per_sample)
        return agg, per_sample, missing

    # Primary eval
    agg, per_sample, missing = evaluate_at_threshold(args.threshold)

    result = {
        "name": args.name,
        "dataset": "binary_mask",
        "threshold": args.threshold,
        "postprocess": args.postprocess,
        "num_gt": len(gt_masks),
        "num_matched": len(per_sample),
        "num_missing": len(missing),
        "missing_keys": missing[:50],
        "metrics": agg,
    }

    print(json.dumps(result["metrics"], indent=2), flush=True)

    # Threshold sweep
    if args.sweep_thresholds:
        thresholds = [float(x.strip()) for x in args.sweep_thresholds.split(",") if x.strip()]
        sweep_rows = []
        for t in thresholds:
            a, _, _ = evaluate_at_threshold(t)
            sweep_rows.append({"threshold": t, **a})
        result["threshold_sweep"] = sweep_rows

        print("\nThreshold sweep:")
        print(f"{'thr':>6}  {'cIoU':>8}  {'gIoU':>8}  {'mDice':>8}  {'P@0.5':>8}  {'P@0.7':>8}")
        for r in sweep_rows:
            print(f"{r['threshold']:6.2f}  {r['cIoU']:8.4f}  {r['gIoU']:8.4f}  {r['mDice']:8.4f}  {r['P@0.5']:8.4f}  {r['P@0.7']:8.4f}")

    # ── write outputs ──────────────────────────────────────────────────────
    # metrics.json
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {metrics_path}")

    # summary.csv
    summary_path = args.output_dir / "summary.csv"
    m = result["metrics"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "num_samples", "cIoU", "gIoU", "mDice", "P@0.5", "P@0.7", "threshold"])
        writer.writeheader()
        writer.writerow({"name": args.name, "num_samples": m["num_samples"],
                         "cIoU": m["cIoU"], "gIoU": m["gIoU"], "mDice": m["mDice"],
                         "P@0.5": m["P@0.5"], "P@0.7": m["P@0.7"], "threshold": args.threshold})
    print(f"Wrote {summary_path}")

    # per_sample.csv
    per_sample_path = args.output_dir / "per_sample.csv"
    pfields = ["sample_id", "iou", "dice", "precision", "recall", "intersection", "union", "pred_area", "gt_area"]
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pfields, extrasaction="ignore")
        writer.writeheader()
        for row in per_sample:
            writer.writerow(row)
    print(f"Wrote {per_sample_path}")

    # threshold_sweep.csv
    if "threshold_sweep" in result:
        sweep_path = args.output_dir / "threshold_sweep.csv"
        sfields = ["threshold", "cIoU", "gIoU", "mDice", "P@0.5", "P@0.7", "num_samples"]
        with open(sweep_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sfields, extrasaction="ignore")
            writer.writeheader()
            for row in result["threshold_sweep"]:
                writer.writerow(row)
        print(f"Wrote {sweep_path}")

    # visuals
    if args.save_visuals:
        vis_dir = args.output_dir / "visuals"
        vis_dir.mkdir(exist_ok=True)
        n_vis = 0
        for key, gt in sorted(gt_masks.items()):
            if n_vis >= args.max_visuals:
                break
            pp = pred_paths.get(key)
            if pp is None:
                continue
            pred = image_to_mask(Image.open(pp), args.threshold)
            if args.postprocess:
                pred = largest_component(pred)
            vis_img = make_visual(pred, gt, key)
            vis_img.save(vis_dir / f"{key}.png")
            n_vis += 1
        print(f"Wrote {n_vis} visuals to {vis_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
