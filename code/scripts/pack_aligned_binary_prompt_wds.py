#!/usr/bin/env python3
"""Pack RefCOCOg / ReasonSeg / SA-CO into geometry-aligned binary-prompt WDS.

Fast implementation: reads only necessary parquet columns, uses pure PIL
polygon fill for mask generation, resolves all image paths in batch mode.

Output WDS schema per sample:
  __key__   — e.g. refcocog_train_00005023
  in.png    — instruction strip + original image
  out.png   — binary mask (255=fg, 0=bg), aligned with image body
  task.txt  — referring/reasoning expression
  meta.json — dataset, image_id, expression, mask_area, etc.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ── dataset registry ──────────────────────────────────────────────────────
REGISTRY = {
    "refcocog": {
        "train_parquet": "/root/private_data/uvlm/refcocog/data/train-00000-of-00001-4fe3e6340cfb69ed.parquet",
        "val_parquet": "/root/private_data/uvlm/refcocog/data/validation-00000-of-00001-15168dfe7b5961e5.parquet",
        "image_root": "/root/private_data/uvlm/instructcv_understanding_flowinone/data/coco2014_local",
        "task_label": "referring_expression_segmentation_binary_mask",
        "strip_title": "Task: referring segmentation",
        "output_hint": "Output: white=target, black=background",
    },
    "reasonseg": {
        "train_parquet": "/root/private_data/uvlm/reasonseg/data/train-00000-of-00001-4fe3e6340cfb69ed.parquet",
        "val_parquet": "/root/private_data/uvlm/reasonseg/data/validation-00000-of-00001-15168dfe7b5961e5.parquet",
        "image_root": "/root/private_data/uvlm/instructcv_understanding_flowinone/data/coco2014_local",
        "task_label": "reasoning_segmentation",
        "strip_title": "Task: reasoning segmentation",
        "output_hint": "Output: white=target, black=background",
    },
    "saco": {
        "train_parquet": None,
        "val_parquet": None,
        "image_root": None,
        "task_label": "concept_segmentation",
        "strip_title": "Task: concept segmentation",
        "output_hint": "Output: white=concept, black=background",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["refcocog", "reasonseg", "saco"])
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples-per-shard", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--parquet", type=Path, default=None)
    p.add_argument("--image-root", type=Path, default=None)
    return p.parse_args()


# ── font ──────────────────────────────────────────────────────────────────
def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    base = "/usr/share/fonts/truetype"
    if bold:
        candidates = [
            f"{base}/dejavu/DejaVuSans-Bold.ttf",
            f"{base}/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            f"{base}/dejavu/DejaVuSans.ttf",
            f"{base}/liberation/LiberationSans-Regular.ttf",
        ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


# ── mask from polygon ──────────────────────────────────────────────────────
def polygon_to_mask(seg, w: int, h: int) -> np.ndarray:
    """COCO segmentation → binary uint8 mask (0/255)."""
    if seg is None or (isinstance(seg, float) and np.isnan(seg)):
        return np.zeros((h, w), dtype=np.uint8)

    if isinstance(seg, list) and len(seg) > 0:
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        for poly in seg:
            if len(poly) < 6:
                continue
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            d.polygon(pts, fill=255)
        return np.asarray(mask, dtype=np.uint8)

    if isinstance(seg, dict):
        counts = seg.get("counts")
        size = seg.get("size", [h, w])
        if counts is not None:
            return _rle_decode(counts, size[1], size[0])  # w, h

    return np.zeros((h, w), dtype=np.uint8)


def _rle_decode(counts, w: int, h: int) -> np.ndarray:
    """Decode RLE counts to binary mask."""
    import re
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    if isinstance(counts, str):
        mask = np.zeros(h * w, dtype=np.uint8)
        pos = 0
        val = 0
        for m in re.finditer(r"\d+", counts):
            run = int(m.group())
            if val:
                mask[pos : pos + run] = 1
            pos += run
            val = 1 - val
        return mask.reshape((h, w), order="F") * 255
    return np.zeros((h, w), dtype=np.uint8)


# ── prompt builder ─────────────────────────────────────────────────────────
def make_aligned_prompt(
    image: Image.Image, expression: str, info: dict, font_scale: float = 1.0
) -> Image.Image:
    """Overlay instruction strip at top of image, keeping body geometry intact."""
    w, h = image.size
    strip_h = max(36, int(round(h * 0.09 * font_scale)))

    canvas = Image.new("RGB", (w, h + strip_h), color=(245, 245, 238))
    canvas.paste(image, (0, strip_h))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Strip background and separator
    draw.rectangle((0, 0, w, strip_h), fill=(245, 245, 238, 235))
    draw.line((0, strip_h - 1, w, strip_h - 1), fill=(20, 20, 20, 200), width=max(1, h // 256))

    title_font = _load_font(max(10, int(round(strip_h * 0.30))), bold=True)
    expr_font = _load_font(max(9, int(round(strip_h * 0.24))), bold=False)
    hint_font = _load_font(max(8, int(round(strip_h * 0.18))), bold=False)

    margin = max(4, w // 100)
    draw.text((margin, max(2, strip_h // 12)), info["strip_title"],
              fill=(12, 12, 12, 255), font=title_font)

    # Truncate long expressions
    max_chars = max(20, w // 7)
    expr = expression.strip()
    if len(expr) > max_chars:
        expr = expr[: max_chars - 3] + "..."

    draw.text((margin, max(strip_h // 2, int(round(strip_h * 0.50)))),
              expr, fill=(60, 60, 60, 255), font=expr_font)
    draw.text((margin, strip_h - int(round(strip_h * 0.28))),
              info["output_hint"], fill=(130, 130, 130, 255), font=hint_font)

    return canvas


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── main processing ────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    info = REGISTRY[args.dataset]
    if args.dataset == "saco":
        print("SA-CO not implemented yet", flush=True)
        return

    # Resolve parquet
    pq_path = str(args.parquet) if args.parquet else info[f"{args.split}_parquet"]
    if not Path(pq_path).exists():
        raise FileNotFoundError(f"Parquet not found: {pq_path}")

    image_root = Path(args.image_root) if args.image_root else Path(info["image_root"])

    print(f"Dataset: {args.dataset}  Split: {args.split}  Image-size: {args.image_size}")
    print(f"Parquet: {pq_path}")
    print(f"Images:  {image_root}")
    print(flush=True)

    # ── Load parquet (specific columns only for speed) ─────────────────────
    cols = ["split", "file_name", "raw_anns", "raw_image_info", "sentences",
            "image_id", "ann_id", "ref_id"]
    df = pd.read_parquet(pq_path, columns=[c for c in cols if c in pd.read_parquet(pq_path).columns])
    print(f"Loaded {len(df)} rows", flush=True)

    # ── Batch-resolve image paths ──────────────────────────────────────────
    print("Resolving image paths...", flush=True)
    # Index all available images
    img_files: dict[str, Path] = {}
    for subdir in ["train2014", "val2014"]:
        sd = image_root / subdir
        if sd.is_dir():
            for fp in sd.iterdir():
                if fp.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    img_files[fp.name] = fp

    # Build image path lookup for each row
    def resolve_img_path(row) -> tuple[Path | None, str]:
        ri = row.get("raw_image_info")
        if isinstance(ri, dict):
            fname = ri.get("file_name", "")
        else:
            try:
                ri = json.loads(ri) if isinstance(ri, str) else {}
                fname = ri.get("file_name", "")
            except Exception:
                fname = ""
        if not fname:
            fname = row.get("file_name", "")
        if fname in img_files:
            return img_files[fname], fname
        # Try glob
        stem = Path(fname).stem
        for k, v in img_files.items():
            if stem in k or k.startswith(stem[:20]):
                return v, fname
        return None, fname

    img_lookup = {}
    for idx, row in df.iterrows():
        p, _ = resolve_img_path(row)
        img_lookup[idx] = p

    n_found = sum(1 for p in img_lookup.values() if p is not None)
    print(f"  Found {n_found}/{len(df)} images", flush=True)

    # ── Prepare output ─────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old in list(args.output_dir.glob("pairs-*.tar")):
            old.unlink(missing_ok=True)

    tar = None
    shard_idx = -1
    yielded = 0
    skipped = 0
    t_start = time.time()

    try:
        for i, (idx, row) in enumerate(df.iterrows()):
            if args.max_samples and yielded >= args.max_samples:
                break

            # ── image ──────────────────────────────────────────────────────
            img_path = img_lookup.get(idx)
            if img_path is None:
                skipped += 1
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            # ── expression ─────────────────────────────────────────────────
            sents = row.get("sentences")
            expression = ""
            if isinstance(sents, (list, np.ndarray)) and len(sents) > 0:
                first = sents[0]
                if isinstance(first, dict):
                    expression = first.get("raw", first.get("sent", ""))
                else:
                    expression = str(first)
            if not expression:
                # Try captions
                caps = row.get("captions")
                if isinstance(caps, (list, np.ndarray)) and len(caps) > 0:
                    expression = str(caps[0])
            if not expression:
                expression = "unknown target"

            # ── mask ───────────────────────────────────────────────────────
            ann = row.get("raw_anns")
            if isinstance(ann, str):
                ann = json.loads(ann)
            if ann is None:
                skipped += 1
                continue
            seg = ann.get("segmentation") if isinstance(ann, dict) else None
            mask = polygon_to_mask(seg, image.width, image.height)

            # ── resize ─────────────────────────────────────────────────────
            # Scale so shorter side = image_size, then center-crop
            orig_w, orig_h = image.size
            scale = args.image_size / min(orig_w, orig_h)
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))

            img_small = image.resize((new_w, new_h), Image.BICUBIC)
            mask_small = Image.fromarray(mask, mode="L").resize((new_w, new_h), Image.NEAREST)

            # Center crop
            if new_w > args.image_size or new_h > args.image_size:
                left = (new_w - args.image_size) // 2 if new_w > args.image_size else 0
                top = (new_h - args.image_size) // 2 if new_h > args.image_size else 0
                img_small = img_small.crop((left, top, left + args.image_size, top + args.image_size))
                mask_small = mask_small.crop((left, top, left + args.image_size, top + args.image_size))

            # ── build prompt ───────────────────────────────────────────────
            prompt = make_aligned_prompt(img_small, expression, info)

            # ── key ────────────────────────────────────────────────────────
            ref_id = row.get("ref_id", row.get("ann_id", i))
            split_name = "train" if str(row.get("split", "train")).lower().startswith("train") else "val"
            key = f"{args.dataset}_{split_name}_{ref_id}"

            # ── sidecar ────────────────────────────────────────────────────
            meta = {
                "dataset": args.dataset,
                "task": info["task_label"],
                "expression": expression,
                "image_id": str(row.get("image_id", "")),
                "ann_id": str(row.get("ann_id", "")),
                "ref_id": str(ref_id),
                "mask_area": int(np.asarray(mask_small).sum() // 255),
                "split": split_name,
            }

            # ── open shard ─────────────────────────────────────────────────
            if yielded % args.samples_per_shard == 0:
                if tar is not None:
                    tar.close()
                shard_idx += 1
                shard_path = args.output_dir / f"pairs-{shard_idx:06d}.tar"
                print(f"[shard {shard_idx}] {shard_path}", flush=True)
                tar = tarfile.open(shard_path, mode="w")

            # ── write ──────────────────────────────────────────────────────
            for fname, fdata in [
                (f"{key}.in.png", png_bytes(prompt)),
                (f"{key}.out.png", png_bytes(mask_small.convert("RGB"))),
                (f"{key}.task.txt", expression.encode("utf-8")),
                (f"{key}.meta.json", json.dumps(meta).encode("utf-8")),
            ]:
                ti = tarfile.TarInfo(fname)
                ti.size = len(fdata)
                tar.addfile(ti, io.BytesIO(fdata))

            yielded += 1

            if yielded % 500 == 0:
                elapsed = time.time() - t_start
                rate = yielded / elapsed if elapsed > 0 else 0
                print(f"  {yielded} samples  ({rate:.1f}/s)  skipped={skipped}", flush=True)
    finally:
        if tar is not None:
            tar.close()

    elapsed = time.time() - t_start
    print(f"\nDone: {yielded} samples, {skipped} skipped, "
          f"{shard_idx + 1} shards in {elapsed:.0f}s → {args.output_dir}", flush=True)

    # Manifest
    manifest = {
        "dataset": args.dataset, "split": args.split,
        "samples": yielded, "skipped": skipped,
        "num_shards": shard_idx + 1, "samples_per_shard": args.samples_per_shard,
        "image_size": args.image_size, "parquet": pq_path,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
