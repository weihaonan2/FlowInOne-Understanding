#!/usr/bin/env python3
"""Pack ReasonSeg dataset into geometry-aligned binary-prompt WDS.

ReasonSeg data format (from /root/private_data/uvlm/ReasonSeg):
  train/val/
    XXXXX.json  — annotation with "text" (reasoning queries), "shapes" (polygon)
    XXXXX.jpg   — source image

Output WDS: in.png / out.png / task.txt / meta.json (aligned binary prompt)
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REASONSEG_ROOT = Path("/root/private_data/uvlm/ReasonSeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples-per-shard", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--data-root", type=Path, default=REASONSEG_ROOT)
    return p.parse_args()


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    base = "/usr/share/fonts/truetype"
    candidates = []
    if bold:
        candidates = [f"{base}/dejavu/DejaVuSans-Bold.ttf",
                       f"{base}/liberation/LiberationSans-Bold.ttf"]
    else:
        candidates = [f"{base}/dejavu/DejaVuSans.ttf",
                       f"{base}/liberation/LiberationSans-Regular.ttf"]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def polygon_points_to_mask(points: list[list[float]], w: int, h: int) -> np.ndarray:
    """Convert list of [x,y] points to binary mask."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) >= 3:
        draw.polygon(pts, fill=255)
    return np.asarray(mask, dtype=np.uint8)


def make_aligned_prompt(image: Image.Image, query: str) -> Image.Image:
    """Overlay reasoning instruction strip on top of image."""
    w, h = image.size
    strip_h = max(42, int(round(h * 0.11)))  # slightly taller for reasoning queries

    canvas = Image.new("RGB", (w, h + strip_h), color=(245, 245, 238))
    canvas.paste(image, (0, strip_h))
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.rectangle((0, 0, w, strip_h), fill=(245, 245, 238, 235))
    draw.line((0, strip_h - 1, w, strip_h - 1), fill=(20, 20, 20, 200), width=max(1, h // 256))

    title_font = _load_font(max(9, int(round(strip_h * 0.25))), bold=True)
    query_font = _load_font(max(8, int(round(strip_h * 0.20))), bold=False)
    hint_font = _load_font(max(7, int(round(strip_h * 0.16))), bold=False)

    margin = max(3, w // 100)
    draw.text((margin, max(1, strip_h // 14)), "Task: reasoning segmentation",
              fill=(12, 12, 12, 255), font=title_font)

    # Truncate query
    max_chars = max(30, w // 5)
    query_short = query.strip()
    if len(query_short) > max_chars:
        query_short = query_short[:max_chars - 3] + "..."

    draw.text((margin, max(strip_h // 2 - 4, int(round(strip_h * 0.42)))),
              query_short, fill=(60, 60, 60, 255), font=query_font)
    draw.text((margin, strip_h - int(round(strip_h * 0.30))),
              "Output: white=target, black=background",
              fill=(130, 130, 130, 255), font=hint_font)

    return canvas


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    args = parse_args()
    src_dir = args.data_root / args.split
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old in list(args.output_dir.glob("pairs-*.tar")):
            old.unlink(missing_ok=True)

    # Collect annotation files
    json_files = sorted(src_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON annotations in {src_dir}", flush=True)

    tar = None
    shard_idx = -1
    yielded = 0
    skipped = 0
    t_start = time.time()

    try:
        for jf in json_files:
            if args.max_samples and yielded >= args.max_samples:
                break

            ann = json.loads(jf.read_text())

            # ── image ──────────────────────────────────────────────────────
            img_name = None
            shapes = ann.get("shapes", [])
            if shapes and len(shapes) > 0:
                img_name = shapes[0].get("image_name", "")
            if not img_name:
                img_name = jf.stem + ".jpg"

            img_path = src_dir / img_name
            if not img_path.exists():
                skipped += 1
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            # ── query ───────────────────────────────────────────────────────
            texts = ann.get("text", [])
            if isinstance(texts, list) and len(texts) > 0:
                query = str(texts[0])
            elif isinstance(texts, str):
                query = texts
            else:
                query = "unknown target"
            # Clean up: first 200 chars
            query = query.strip()[:200]

            # ── mask ───────────────────────────────────────────────────────
            # Combine all polygon shapes
            full_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            for shape in shapes:
                points = shape.get("points", [])
                if points and len(points) >= 3:
                    mask_part = polygon_points_to_mask(points, image.width, image.height)
                    full_mask = np.maximum(full_mask, mask_part)

            # ── resize ─────────────────────────────────────────────────────
            orig_w, orig_h = image.size
            scale = args.image_size / min(orig_w, orig_h)
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))

            img_small = image.resize((new_w, new_h), Image.BICUBIC)
            mask_small = Image.fromarray(full_mask, mode="L").resize((new_w, new_h), Image.NEAREST)

            if new_w > args.image_size or new_h > args.image_size:
                left = (new_w - args.image_size) // 2 if new_w > args.image_size else 0
                top = (new_h - args.image_size) // 2 if new_h > args.image_size else 0
                img_small = img_small.crop((left, top, left + args.image_size, top + args.image_size))
                mask_small = mask_small.crop((left, top, left + args.image_size, top + args.image_size))

            # ── prompt ─────────────────────────────────────────────────────
            prompt = make_aligned_prompt(img_small, query)

            # ── key ────────────────────────────────────────────────────────
            key = f"reasonseg_{args.split}_{jf.stem}"

            # ── meta ───────────────────────────────────────────────────────
            meta = {
                "dataset": "reasonseg",
                "task": "reasoning_segmentation",
                "query": query,
                "image_name": img_name,
                "mask_area": int(np.asarray(mask_small).sum() // 255),
                "split": args.split,
                "num_queries": len(texts) if isinstance(texts, list) else 1,
            }

            # ── shard ──────────────────────────────────────────────────────
            if yielded % args.samples_per_shard == 0:
                if tar is not None:
                    tar.close()
                shard_idx += 1
                shard_path = args.output_dir / f"pairs-{shard_idx:06d}.tar"
                print(f"[shard {shard_idx}] {shard_path}", flush=True)
                tar = tarfile.open(shard_path, mode="w")

            for fname, fdata in [
                (f"{key}.in.png", png_bytes(prompt)),
                (f"{key}.out.png", png_bytes(mask_small.convert("RGB"))),
                (f"{key}.task.txt", query.encode("utf-8")),
                (f"{key}.meta.json", json.dumps(meta).encode("utf-8")),
            ]:
                ti = tarfile.TarInfo(fname)
                ti.size = len(fdata)
                tar.addfile(ti, io.BytesIO(fdata))

            yielded += 1

    finally:
        if tar is not None:
            tar.close()

    elapsed = time.time() - t_start
    print(f"\nDone: {yielded} samples, {skipped} skipped, "
          f"{shard_idx + 1} shards in {elapsed:.0f}s → {args.output_dir}", flush=True)

    manifest = {
        "dataset": "reasonseg", "split": args.split,
        "samples": yielded, "skipped": skipped,
        "num_shards": shard_idx + 1, "samples_per_shard": args.samples_per_shard,
        "image_size": args.image_size, "source": str(args.data_root),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
