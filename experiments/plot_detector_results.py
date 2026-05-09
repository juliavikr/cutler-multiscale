#!/usr/bin/env python
"""Generate a compact PNG comparison for the main 5-class detector results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = "#f6f8fb"
CARD = "#ffffff"
TEXT = "#18324a"
MUTED = "#5b6f82"
GRID = "#dbe4ee"
BLUE = "#3b82f6"
TEAL = "#14b8a6"
ORANGE = "#f59e0b"
GREEN = "#22c55e"
RED = "#ef4444"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="results/detector_results_5class.csv",
        help="Path to the 5-class detector comparison CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/figures/detector_results_5class.png",
        help="Where to save the generated PNG plot.",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(["arialbd.ttf", "DejaVuSans-Bold.ttf"])
    candidates.extend(["arial.ttf", "DejaVuSans.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_11 = load_font(11)
FONT_12 = load_font(12)
FONT_13 = load_font(13)
FONT_16 = load_font(16, bold=True)
FONT_18 = load_font(18, bold=True)
FONT_28 = load_font(28, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_centered(draw, x, y, text, font, fill=TEXT):
    w, h = text_size(draw, text, font)
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def draw_right(draw, x, y, text, font, fill=TEXT):
    w, h = text_size(draw, text, font)
    draw.text((x - w, y - h / 2), text, font=font, fill=fill)


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label(name: str) -> str:
    mapping = {
        "baseline_single_scale": "baseline",
        "old_multiscale_only": "old multiscale",
        "old_combined": "old combined",
        "new_hybrid_only": "new hybrid",
        "new_combined_hybrid_best": "new combined",
    }
    return mapping.get(name, name.replace("_", " "))


def main():
    args = parse_args()
    rows = load_rows(Path(args.csv))

    width, height = 1280, 760
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    draw_centered(draw, width / 2, 40, "Final 5-Class Detector Results", FONT_28, TEXT)
    draw_centered(draw, width / 2, 72, "Class-agnostic COCO evaluation", FONT_13, MUTED)

    cards = [
        ("Best bbox AP", "2.2557", BLUE),
        ("Best segm AP", "1.0814", TEAL),
        ("Baseline bbox AP", "2.1447", ORANGE),
        ("Baseline segm AP", "0.4792", GREEN),
    ]
    x = 48
    for title, value, color in cards:
        draw.rounded_rectangle((x, 110, x + 270, 220), radius=18, fill=CARD, outline="#e7edf4")
        draw.text((x + 20, 132), title, font=FONT_16, fill=TEXT)
        draw.text((x + 20, 170), value, font=FONT_28, fill=color)
        x += 294

    plot_x0, plot_y0 = 120, 670
    plot_w, plot_h = 1040, 350
    y_max = 6.0
    draw.line((plot_x0, plot_y0 - plot_h, plot_x0, plot_y0), fill=TEXT, width=2)
    draw.line((plot_x0, plot_y0, plot_x0 + plot_w, plot_y0), fill=TEXT, width=2)
    for i in range(7):
        value = y_max * i / 6
        y = plot_y0 - plot_h * i / 6
        draw.line((plot_x0, y, plot_x0 + plot_w, y), fill=GRID, width=1)
        draw_right(draw, plot_x0 - 10, y, f"{value:.1f}", FONT_11, MUTED)

    group_w = plot_w / len(rows)
    bar_w = group_w * 0.22
    for idx, row in enumerate(rows):
        cx = plot_x0 + group_w * (idx + 0.5)
        bbox_ap = float(row["bbox_ap"])
        segm_ap = float(row["segm_ap"])
        for value, color, offset in [(bbox_ap, BLUE, -bar_w * 0.7), (segm_ap, TEAL, bar_w * 0.1)]:
            h = plot_h * value / y_max
            outline = RED if row["variant"] == "new_combined_hybrid_best" else None
            draw.rounded_rectangle(
                (cx + offset, plot_y0 - h, cx + offset + bar_w, plot_y0),
                radius=5,
                fill=color,
                outline=outline,
                width=3 if outline else 0,
            )
            draw_centered(draw, cx + offset + bar_w / 2, plot_y0 - h - 14, f"{value:.2f}", FONT_11, TEXT)
        draw_centered(draw, cx, plot_y0 + 30, label(row["variant"]), FONT_12, MUTED)

    draw.rounded_rectangle((940, 270, 1180, 340), radius=14, fill=CARD, outline="#e7edf4")
    draw.rounded_rectangle((962, 288, 976, 302), radius=3, fill=BLUE)
    draw.text((986, 282), "bbox AP", font=FONT_12, fill=MUTED)
    draw.rounded_rectangle((962, 316, 976, 330), radius=3, fill=TEAL)
    draw.text((986, 310), "segm AP", font=FONT_12, fill=MUTED)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
