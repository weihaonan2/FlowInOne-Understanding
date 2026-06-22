#!/usr/bin/env python3
"""Evaluate RefCOCOg/ReasonSeg checkpoints: generate predictions + compute metrics.

Usage:
  python eval_aligned_binary_checkpoints.py \
    --config CONFIG.py \
    --ckpt-dir CKPT_DIR/ \
    --val-wds VAL_WDS_PATTERN \
    --output-dir EVAL_OUTPUT/ \
    --task [refcocog|reasonseg]

For each checkpoint in CKPT_DIR (e.g. 250.ckpt, 500.ckpt...):
  1. Extract val input images from WDS
  2. Run i2i inference to generate predictions
  3. Compute cIoU, gIoU, mDice, P@0.5, P@0.7
  4. Save metrics.json, per_sample.csv, summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--val-wds", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", default="refcocog")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sweep-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--steps", default=None,
                   help="Comma-separated checkpoint step numbers (default: auto-detect)")
    p.add_argument("--cfg", type=float, default=7.0)
    p.add_argument("--no-inference", action="store_true",
                   help="Skip inference, only compute metrics on existing preds")
    return p.parse_args()


def expand(pattern: str) -> list[Path]:
    try:
        import braceexpand
        return [Path(p) for p in sorted(braceexpand.braceexpand(pattern))]
    except Exception:
        return sorted(Path(p) for p in glob.glob(pattern))


def extract_val_inputs(wds_pattern: str, out_dir: Path, max_samples: int | None) -> Path:
    """Extract in.png files from val WDS into out_dir/input/ and out_dir/gt/."""
    input_dir = out_dir / "input"
    gt_dir = out_dir / "gt"
    input_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for tar_path in expand(wds_pattern):
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if name.endswith(".in.png"):
                    key = Path(name).name.rsplit(".", 2)[0]
                    f = tar.extractfile(member)
                    if f:
                        (input_dir / f"{key}.png").write_bytes(f.read())
                elif name.endswith(".out.png"):
                    key = Path(name).name.rsplit(".", 2)[0]
                    f = tar.extractfile(member)
                    if f:
                        (gt_dir / f"{key}.png").write_bytes(f.read())
                        count += 1
                        if max_samples and count >= max_samples:
                            break
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break

    print(f"Extracted {count} val pairs to {out_dir}")
    return out_dir


def run_inference(config: Path, nnet_path: Path, input_dir: Path, output_dir: Path,
                  cfg: float = 7.0) -> bool:
    """Run i2i inference for a single checkpoint."""
    i2i_script = ROOT / "instructcv_understanding_flowinone" / "scripts" / "run_flowinone_i2i_direct_x1_no_flash.py"

    if not i2i_script.exists():
        print(f"ERROR: Inference script not found: {i2i_script}", file=sys.stderr)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(i2i_script),
        f"--config={config}",
        f"--nnet_path={nnet_path}",
        f"--input_image_path={input_dir}",
        f"--output_image_path={output_dir}",
        f"--cfg={cfg}",
        "--direct_t=0",
        "--batch_size=2",
    ]

    print(f"  Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"})

    if result.returncode != 0:
        print(f"  Inference FAILED:\n{result.stderr[-500:]}", file=sys.stderr)
        return False
    print(f"  Inference OK, output in {output_dir}")
    return True


def compute_metrics(pred_dir: Path, gt_dir: Path, output_dir: Path,
                    threshold: float, sweep_thresholds: str | None,
                    name: str) -> dict:
    """Run eval_binary_mask_metrics.py."""
    eval_script = ROOT / "teacher_understanding_eval" / "scripts" / "eval_binary_mask_metrics.py"

    cmd = [
        sys.executable, str(eval_script),
        f"--pred-dir={pred_dir}",
        f"--gt-dir={gt_dir}",
        f"--output-dir={output_dir}",
        f"--threshold={threshold}",
        f"--name={name}",
    ]
    if sweep_thresholds:
        cmd.append(f"--sweep-thresholds={sweep_thresholds}")

    print(f"  Computing metrics: {name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  Metrics FAILED:\n{result.stderr[-500:]}", file=sys.stderr)
        return {}

    # Parse metrics.json
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return {}


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract val data ───────────────────────────────────────────────────
    val_dir = args.output_dir / "val_data"
    if not (val_dir / "input").exists():
        print("Extracting val data...")
        extract_val_inputs(args.val_wds, val_dir, args.max_samples)
    else:
        print(f"Val data already exists at {val_dir}")

    input_dir = val_dir / "input"
    gt_dir = val_dir / "gt"

    # ── Find checkpoints ───────────────────────────────────────────────────
    if args.steps:
        steps_raw = [s.strip() for s in args.steps.split(",")]
        steps = []
        for s in steps_raw:
            try:
                steps.append(int(s))
            except ValueError:
                # Non-numeric step name (e.g. "original") — use as-is via dir name
                steps.append(s)
    else:
        steps = []
        for d in sorted(args.ckpt_dir.iterdir()):
            if d.is_dir() and d.name.endswith(".ckpt"):
                try:
                    steps.append(int(d.name.replace(".ckpt", "")))
                except ValueError:
                    pass
        steps = sorted(steps)

    print(f"\nFound {len(steps)} checkpoints: {steps}")
    print(f"Config: {args.config}")
    print(f"Val data: {val_dir}")
    print()

    # ── Evaluate each checkpoint ───────────────────────────────────────────
    all_results = []
    for step in steps:
        ckpt_dir = args.ckpt_dir / f"{step}.ckpt"
        nnet_path = ckpt_dir / "nnet.pth"
        if not nnet_path.exists():
            print(f"  Skip {step}.ckpt — no nnet.pth")
            continue

        name = f"{args.task}_step{step}"
        pred_dir = args.output_dir / f"pred_step{step}"
        eval_dir = args.output_dir / f"eval_step{step}"

        print(f"\n{'='*60}")
        print(f"Checkpoint: {step}.ckpt")
        print(f"{'='*60}")

        if not args.no_inference:
            ok = run_inference(args.config, nnet_path, input_dir, pred_dir, args.cfg)
            if not ok:
                continue

        metrics = compute_metrics(pred_dir, gt_dir, eval_dir,
                                  args.threshold, args.sweep_thresholds, name)
        if metrics:
            m = metrics.get("metrics", {})
            print(f"  cIoU={m.get('cIoU', 0):.4f}  gIoU={m.get('gIoU', 0):.4f}  "
                  f"mDice={m.get('mDice', 0):.4f}  P@0.5={m.get('P@0.5', 0):.4f}")
            all_results.append({"step": step, **m})

    # ── Summary table ──────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Step':>8}  {'cIoU':>8}  {'gIoU':>8}  {'mDice':>8}  {'P@0.5':>8}  {'P@0.7':>8}")
        for r in all_results:
            print(f"{r['step']:8d}  {r.get('cIoU', 0):8.4f}  {r.get('gIoU', 0):8.4f}  "
                  f"{r.get('mDice', 0):8.4f}  {r.get('P@0.5', 0):8.4f}  {r.get('P@0.7', 0):8.4f}")

        # Save summary
        summary_path = args.output_dir / "all_metrics_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "cIoU", "gIoU", "mDice", "P@0.5", "P@0.7"])
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k, 0) for k in writer.fieldnames})
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
