#!/usr/bin/env python3
"""Pack Cityscapes into geometry-aligned 7-group semantic WDS pairs.

This is different from Cityscapes gtCoarse. It keeps the same gtFine samples but
changes the output language from 19 trainId colors to 7 coarse semantic groups:
flat / construction / object / nature / sky / human / vehicle.

The input prompt is also rewritten with a compact 7-group legend so the model is
not asked to generate the original 19-class palette.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import braceexpand
import numpy as np
import webdataset as wds
from PIL import Image, ImageDraw, ImageFont


IGNORE_TRAIN_ID = 255

CLASS_NAMES_19 = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

TRAIN_ID_TO_COLOR_19 = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.float32,
)

COARSE7_CLASSES: list[tuple[str, tuple[int, int, int]]] = [
    ("flat", (128, 64, 128)),          # road / sidewalk
    ("construction", (70, 70, 70)),    # building / wall / fence
    ("object", (153, 153, 153)),       # pole / traffic light / traffic sign
    ("nature", (107, 142, 35)),        # vegetation / terrain
    ("sky", (70, 130, 180)),
    ("human", (220, 20, 60)),          # person / rider
    ("vehicle", (0, 0, 142)),          # car / truck / bus / train / motorcycle / bicycle
]

COARSE7_COLOR_TABLE = np.asarray([color for _, color in COARSE7_CLASSES], dtype=np.uint8)

TRAIN_ID_TO_COARSE7 = np.asarray(
    [
        0,  # road -> flat
        0,  # sidewalk -> flat
        1,  # building -> construction
        1,  # wall -> construction
        1,  # fence -> construction
        2,  # pole -> object
        2,  # traffic light -> object
        2,  # traffic sign -> object
        3,  # vegetation -> nature
        3,  # terrain -> nature
        4,  # sky -> sky
        5,  # person -> human
        5,  # rider -> human
        6,  # car -> vehicle
        6,  # truck -> vehicle
        6,  # bus -> vehicle
        6,  # train -> vehicle
        6,  # motorcycle -> vehicle
        6,  # bicycle -> vehicle
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tars", required=True, help="Input 19-class Cityscapes WDS tar pattern, supports brace expansion.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def scaled_font(image: Image.Image, fraction: float, bold: bool = False) -> ImageFont.ImageFont:
    return load_font(max(10, round(image.height * fraction)), bold=bold)


def rgb_to_train_ids_nearest(mask_rgb: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(mask_rgb.convert("RGB"), dtype=np.float32)
    diff = arr[:, :, None, :] - TRAIN_ID_TO_COLOR_19[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    train_ids = np.argmin(dist2, axis=-1).astype(np.uint8)
    ignore = np.all(arr == 0, axis=-1)
    return train_ids, ignore


def mask19_to_coarse7(mask: Image.Image) -> Image.Image:
    train_ids, ignore = rgb_to_train_ids_nearest(mask)
    coarse = TRAIN_ID_TO_COARSE7[train_ids]
    rgb = COARSE7_COLOR_TABLE[coarse]
    rgb[ignore] = (0, 0, 0)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def make_coarse7_aligned_prompt(image: Image.Image) -> Image.Image:
    prompt = image.convert("RGB").copy()
    draw = ImageDraw.Draw(prompt, "RGBA")
    width, height = prompt.size

    band_h = max(48, round(height * 0.105))
    draw.rectangle((0, 0, width, band_h), fill=(245, 245, 238, 222))
    draw.line((0, band_h, width, band_h), fill=(20, 20, 20, 210), width=max(1, height // 256))

    title_font = scaled_font(prompt, 0.028, bold=True)
    small_font = scaled_font(prompt, 0.019)
    title = "Cityscapes 7-group semantic layout: output only the coarse color mask"
    draw.text((max(8, width // 128), max(4, band_h // 12)), title, fill=(12, 12, 12, 255), font=title_font)

    swatch = max(8, round(height * 0.022))
    gap = max(5, round(width * 0.006))
    x = max(8, width // 128)
    y = max(band_h // 2, round(height * 0.054))
    for name, color in COARSE7_CLASSES:
        if x + swatch + 2 * gap + 120 > width:
            break
        draw.rectangle((x, y, x + swatch, y + swatch), fill=(*color, 255), outline=(0, 0, 0, 230))
        draw.text((x + swatch + gap, y - max(1, swatch // 8)), name, fill=(12, 12, 12, 255), font=small_font)
        text_w = draw.textlength(name, font=small_font)
        x += swatch + gap + round(text_w) + 3 * gap
    return prompt


def iter_pairs(input_tars: str):
    for tar_path in braceexpand.braceexpand(input_tars):
        dataset = wds.WebDataset(tar_path).decode("pil")
        for sample in dataset:
            if "in.png" not in sample or "out.png" not in sample:
                continue
            yield sample["__key__"], sample["in.png"], sample["out.png"]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old_file in args.output_dir.glob("pairs-*.tar"):
            old_file.unlink()

    sink = None
    shard_id = -1
    count = 0
    try:
        for key, image, mask in iter_pairs(args.input_tars):
            if args.max_samples is not None and count >= args.max_samples:
                break

            if count % args.samples_per_shard == 0:
                if sink is not None:
                    sink.close()
                shard_id += 1
                tar_path = args.output_dir / f"pairs-{shard_id:06d}.tar"
                print(f"open {tar_path}", flush=True)
                sink = wds.TarWriter(str(tar_path))

            assert sink is not None
            prompt = make_coarse7_aligned_prompt(image)
            coarse_mask = mask19_to_coarse7(mask)
            sink.write({"__key__": key, "in.png": png_bytes(prompt), "out.png": png_bytes(coarse_mask)})
            count += 1
            if count % 100 == 0:
                print(f"packed {count} samples", flush=True)
    finally:
        if sink is not None:
            sink.close()

    print(f"packed {count} samples into {shard_id + 1 if count else 0} shards at {args.output_dir}")


if __name__ == "__main__":
    main()
