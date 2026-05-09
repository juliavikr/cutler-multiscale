#!/usr/bin/env python
"""Generate PNG plots for the 100-image hybrid ablation summary."""

from __future__ import annotations

import argparse
import csv
import math
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
PURPLE = "#8b5cf6"
SLATE = "#94a3b8"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="results/hybrid_ablation_100_summary.csv",
        help="Path to ablation summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/visualizations/hybrid_ablation_summary",
        help="Directory where plots will be saved",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"variant": row["variant"]}
            for key, value in row.items():
                if key == "variant":
                    continue
                parsed[key] = int(value)
            rows.append(parsed)
    return rows


def nice_max(value: float) -> int:
    if value <= 0:
        return 1
    magnitude = 10 ** int(math.floor(math.log10(value)))
    scaled = value / magnitude
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return int(nice * magnitude)


def format_variant(variant: str) -> str:
    mapping = {
        "baseline": "baseline (hp85)",
        "hp90": "hp90",
        "hp80": "hp80",
        "topk8": "topk8",
        "tightcrop": "tightcrop",
    }
    return mapping.get(variant, variant)


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


FONT_10 = load_font(10)
FONT_12 = load_font(12)
FONT_13 = load_font(13)
FONT_14 = load_font(14)
FONT_15 = load_font(15)
FONT_16 = load_font(16)
FONT_18 = load_font(18)
FONT_20 = load_font(20, bold=True)
FONT_28 = load_font(28, bold=True)
FONT_30 = load_font(30, bold=True)
FONT_48 = load_font(48, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_centered(draw, x, y, text, font, fill=TEXT):
    w, h = text_size(draw, text, font)
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def draw_right(draw, x, y, text, font, fill=TEXT):
    w, h = text_size(draw, text, font)
    draw.text((x - w, y - h / 2), text, font=font, fill=fill)


def new_canvas(width: int, height: int):
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    return image, draw


def save_png(image: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def draw_title(draw, width: int, title: str, subtitle: str | None = None):
    draw_centered(draw, width / 2, 42, title, FONT_30, TEXT)
    if subtitle:
        draw_centered(draw, width / 2, 74, subtitle, FONT_15, MUTED)


def draw_legend(draw, x: int, y: int, items: list[tuple[str, str]]):
    for idx, (label, color) in enumerate(items):
        yi = y + idx * 24
        draw.rounded_rectangle((x, yi - 10, x + 14, yi + 4), radius=3, fill=color)
        draw.text((x + 22, yi - 12), label, font=FONT_12, fill=MUTED)


def draw_axes(draw, x0, y0, plot_w, plot_h, y_max, ticks=5):
    draw.line((x0, y0 - plot_h, x0, y0), fill=TEXT, width=2)
    draw.line((x0, y0, x0 + plot_w, y0), fill=TEXT, width=2)
    for i in range(ticks + 1):
        value = y_max * i / ticks
        y = y0 - plot_h * i / ticks
        draw.line((x0, y, x0 + plot_w, y), fill=GRID, width=1)
        draw_right(draw, x0 - 10, y, str(int(round(value))), FONT_12, MUTED)


def save_final_mask_chart(rows, out_dir: Path):
    width, height = 1120, 620
    x0, y0 = 110, 520
    plot_w, plot_h = 970, 370
    y_max = nice_max(max(row["merged"] for row in rows) * 1.12)

    image, draw = new_canvas(width, height)
    draw_title(draw, width, "Final Pseudo-Labels per Variant")
    draw_axes(draw, x0, y0, plot_w, plot_h, y_max)
    draw_legend(draw, 780, 92, [("Full-image masks", BLUE), ("Added crop masks", TEAL)])

    group_w = plot_w / len(rows)
    bar_w = group_w * 0.55
    for i, row in enumerate(rows):
        cx = x0 + group_w * (i + 0.5)
        full = row["full_masks"]
        added = row["merged"] - full
        total = row["merged"]
        full_h = plot_h * full / y_max
        add_h = plot_h * added / y_max
        full_y = y0 - full_h
        add_y = full_y - add_h

        outline = None
        if row["variant"] == "topk8":
            outline = GREEN
        elif row["variant"] == "tightcrop":
            outline = ORANGE

        draw.rounded_rectangle((cx - bar_w / 2, full_y, cx + bar_w / 2, y0), radius=6, fill=BLUE, outline=outline, width=3 if outline else 0)
        draw.rounded_rectangle((cx - bar_w / 2, add_y, cx + bar_w / 2, full_y), radius=6, fill=TEAL, outline=outline, width=3 if outline else 0)
        draw_centered(draw, cx, add_y - 16, str(total), FONT_13, TEXT)
        draw_centered(draw, cx, y0 + 28, format_variant(row["variant"]), FONT_12, MUTED)

    save_png(image, out_dir / "ablation_final_masks_stacked.png")


def save_efficiency_chart(rows, out_dir: Path):
    width, height = 1120, 620
    x0, y0 = 110, 520
    plot_w, plot_h = 970, 370
    crop_keep_ratio = [100.0 * row["crop_merged"] / max(row["generated"], 1) for row in rows]
    final_per_window = [100.0 * row["merged"] / max(row["windows"], 1) for row in rows]
    y_max = nice_max(max(max(crop_keep_ratio), max(final_per_window)) * 1.15)

    image, draw = new_canvas(width, height)
    draw_title(draw, width, "Ablation Efficiency")
    draw_axes(draw, x0, y0, plot_w, plot_h, y_max)
    draw_legend(draw, 720, 92, [("Crop masks kept / generated", ORANGE), ("Final masks / windows", GREEN)])

    group_w = plot_w / len(rows)
    bar_w = group_w * 0.24
    for i, row in enumerate(rows):
        cx = x0 + group_w * (i + 0.5)
        vals = [
            100.0 * row["crop_merged"] / max(row["generated"], 1),
            100.0 * row["merged"] / max(row["windows"], 1),
        ]
        colors = [ORANGE, GREEN]
        offsets = [-bar_w * 0.7, bar_w * 0.1]
        for value, color, offset in zip(vals, colors, offsets):
            h = plot_h * value / y_max
            draw.rounded_rectangle((cx + offset, y0 - h, cx + offset + bar_w, y0), radius=4, fill=color)
            draw_centered(draw, cx + offset + bar_w / 2, y0 - h - 14, f"{value:.1f}%", FONT_10, TEXT)
        draw_centered(draw, cx, y0 + 28, format_variant(row["variant"]), FONT_12, MUTED)

    save_png(image, out_dir / "ablation_efficiency.png")


def save_crop_budget_chart(rows, out_dir: Path):
    width, height = 1120, 620
    x0, y0 = 110, 520
    plot_w, plot_h = 970, 370
    y_max = nice_max(max(max(row["windows"] for row in rows), max(row["crop_merged"] for row in rows), max(row["rescue"] for row in rows)) * 1.12)

    image, draw = new_canvas(width, height)
    draw_title(draw, width, "Crop Budget vs Surviving Crop Masks")
    draw_axes(draw, x0, y0, plot_w, plot_h, y_max)
    draw_legend(draw, 680, 92, [("Crop windows", PURPLE), ("Merged crop masks", SLATE), ("Spatial rescue count", RED)])

    group_w = plot_w / len(rows)
    bar_w = group_w * 0.22
    points = []
    for i, row in enumerate(rows):
        cx = x0 + group_w * (i + 0.5)
        windows = row["windows"]
        crop_merged = row["crop_merged"]
        rescue = row["rescue"]
        for value, color, offset in [(windows, PURPLE, -bar_w * 0.7), (crop_merged, SLATE, bar_w * 0.1)]:
            h = plot_h * value / y_max
            draw.rounded_rectangle((cx + offset, y0 - h, cx + offset + bar_w, y0), radius=4, fill=color)
        py = y0 - plot_h * rescue / y_max
        points.append((cx, py, rescue))
        draw_centered(draw, cx, y0 + 28, format_variant(row["variant"]), FONT_12, MUTED)

    for idx in range(len(points) - 1):
        x1, y1, _ = points[idx]
        x2, y2, _ = points[idx + 1]
        draw.line((x1, y1, x2, y2), fill=RED, width=3)
    for x, y, value in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=RED)
        draw_centered(draw, x, y - 16, str(value), FONT_10, TEXT)

    save_png(image, out_dir / "ablation_crop_budget.png")


def save_delta_chart(rows, out_dir: Path):
    width, height = 1020, 520
    x_left, x_right = 180, 940
    center_x = (x_left + x_right) / 2
    usable = (x_right - x_left) / 2 - 30
    baseline_total = next(row["merged"] for row in rows if row["variant"] == "baseline")
    deltas = [row["merged"] - baseline_total for row in rows]
    max_abs = max(abs(v) for v in deltas) or 1

    image, draw = new_canvas(width, height)
    draw_title(draw, width, "Change vs baseline (hp85)", f"Final merged pseudo-label count relative to baseline = {baseline_total}")
    draw.line((center_x, 96, center_x, height - 60), fill=TEXT, width=2)

    row_h = 70
    start_y = 120
    for i, row in enumerate(rows):
        y = start_y + i * row_h
        variant = row["variant"]
        delta = row["merged"] - baseline_total
        color = BLUE if delta == 0 else (GREEN if delta < 0 else ORANGE)
        draw.text((34, y - 10), format_variant(variant), font=FONT_14, fill=TEXT)
        if delta == 0:
            draw.ellipse((center_x - 8, y - 8, center_x + 8, y + 8), fill=color)
            draw.text((center_x + 16, y - 10), "no change", font=FONT_13, fill=TEXT)
        else:
            bar_w = usable * abs(delta) / max_abs
            if delta > 0:
                x1, x2 = center_x, center_x + bar_w
                tx = x2 + 10
            else:
                x1, x2 = center_x - bar_w, center_x
                tx = x1 - 10
            draw.rounded_rectangle((x1, y - 16, x2, y + 16), radius=8, fill=color)
            label = f"{delta:+d}"
            if delta > 0:
                draw.text((tx, y - 10), label, font=FONT_13, fill=TEXT)
            else:
                draw_right(draw, tx, y, label, FONT_13, TEXT)

    draw_right(draw, center_x - 10, height - 28, "fewer masks", FONT_12, MUTED)
    draw.text((center_x + 10, height - 36), "more masks", font=FONT_12, fill=MUTED)
    save_png(image, out_dir / "ablation_delta_vs_baseline.png")


def save_focus_chart(rows, out_dir: Path):
    focus = [row for row in rows if row["variant"] in {"baseline", "topk8", "tightcrop"}]
    order = {"baseline": 0, "topk8": 1, "tightcrop": 2}
    focus.sort(key=lambda row: order[row["variant"]])

    width, height = 1240, 760
    x0, y0 = 110, 640
    plot_w, plot_h = 1040, 360
    y_max = nice_max(max(row["merged"] for row in focus) * 1.15)

    image, draw = new_canvas(width, height)
    draw_title(
        draw,
        width,
        "Ablation focus: the three meaningful variants",
        "Percentile changes had no visible effect; crop budget and crop size did.",
    )
    draw_axes(draw, x0, y0, plot_w, plot_h, y_max)
    draw_legend(draw, 930, 118, [("Full-image masks", BLUE), ("Added crop masks", TEAL)])

    bullet_lines = [
        (BLUE, "baseline: balanced reference setting"),
        (GREEN, "topk8: fewer windows, cleaner outputs, -23% final masks"),
        (ORANGE, "tightcrop: smaller crops, denser outputs, +65% final masks"),
    ]
    for idx, (color, text) in enumerate(bullet_lines):
        y = 126 + idx * 24
        draw.ellipse((140 - 5, y - 5, 140 + 5, y + 5), fill=color)
        draw.text((154, y - 10), text, font=FONT_13, fill=MUTED)

    group_w = plot_w / len(focus)
    bar_w = group_w * 0.42
    baseline_total = focus[0]["merged"]
    for i, row in enumerate(focus):
        cx = x0 + group_w * (i + 0.5)
        full = row["full_masks"]
        added = row["merged"] - full
        total = row["merged"]
        full_h = plot_h * full / y_max
        add_h = plot_h * added / y_max
        full_y = y0 - full_h
        add_y = full_y - add_h
        outline = BLUE if row["variant"] == "baseline" else (GREEN if row["variant"] == "topk8" else ORANGE)

        draw.rounded_rectangle((cx - bar_w / 2, full_y, cx + bar_w / 2, y0), radius=8, fill=BLUE, outline=outline, width=3)
        draw.rounded_rectangle((cx - bar_w / 2, add_y, cx + bar_w / 2, full_y), radius=8, fill=TEAL, outline=outline, width=3)
        draw_centered(draw, cx, add_y - 18, str(total), FONT_16, TEXT)
        draw_centered(draw, cx, y0 + 34, format_variant(row["variant"]), FONT_15, TEXT)
        delta = total - baseline_total
        delta_text = "reference" if row["variant"] == "baseline" else f"{delta:+d} vs baseline"
        draw_centered(draw, cx, y0 + 58, delta_text, FONT_12, MUTED)

    save_png(image, out_dir / "ablation_focus.png")


def save_dashboard(rows, out_dir: Path):
    baseline = next(row for row in rows if row["variant"] == "baseline")
    topk8 = next(row for row in rows if row["variant"] == "topk8")
    tightcrop = next(row for row in rows if row["variant"] == "tightcrop")

    width, height = 1280, 860
    image, draw = new_canvas(width, height)
    draw_title(
        draw,
        width,
        "Hybrid ablation summary (100-image subset)",
        "The meaningful levers were crop budget and crop size, not heatmap percentile.",
    )

    cards = [
        (48, 112, 370, 180, "Main finding", "Heatmap percentile had no measurable effect here."),
        (454, 112, 370, 180, "Conservative option", "Fewer windows, fewer surviving crop masks, cleaner visual result."),
        (860, 112, 370, 180, "Aggressive option", "Smaller crops raise local sensitivity, but produce many more fragments."),
    ]
    for x, y, w, h, title, subtitle in cards:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=CARD, outline="#e7edf4")
        draw.text((x + 24, y + 18), title, font=FONT_20, fill=TEXT)
        draw.text((x + 24, y + 48), subtitle, font=FONT_13, fill=MUTED)

    draw.text((76, 172), "496", font=FONT_48, fill=BLUE)
    draw.text((176, 180), "final masks for hp80 / hp85 / hp90", font=FONT_18, fill=TEXT)
    draw.text((76, 224), "Identical outputs across all percentile settings.", font=FONT_14, fill=MUTED)

    draw.text((482, 172), str(topk8["merged"]), font=FONT_48, fill=GREEN)
    draw.text((582, 180), "final masks with topk8", font=FONT_18, fill=TEXT)
    draw.text((482, 224), f"-23% vs baseline, using only {topk8['windows']} windows.", font=FONT_14, fill=MUTED)

    draw.text((888, 172), str(tightcrop["merged"]), font=FONT_48, fill=ORANGE)
    draw.text((988, 180), "final masks with tightcrop", font=FONT_18, fill=TEXT)
    draw.text((888, 224), "+65% vs baseline with nearly 2x surviving crop masks.", font=FONT_14, fill=MUTED)

    draw.text((88, 336), "Recommendation:", font=FONT_20, fill=TEXT)
    draw.text((88, 370), "Use topk8 when you want cleaner pseudo-labels. Keep tightcrop as the high-recall but noisy variant.", font=FONT_14, fill=MUTED)
    draw.text((88, 400), f"Baseline merged masks: {baseline['merged']}   |   topk8: {topk8['merged']}   |   tightcrop: {tightcrop['merged']}", font=FONT_14, fill=MUTED)

    save_png(image, out_dir / "ablation_dashboard.png")


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    save_final_mask_chart(rows, out_dir)
    save_efficiency_chart(rows, out_dir)
    save_crop_budget_chart(rows, out_dir)
    save_delta_chart(rows, out_dir)
    save_focus_chart(rows, out_dir)
    save_dashboard(rows, out_dir)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
