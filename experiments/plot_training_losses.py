#!/usr/bin/env python
"""Plot Detectron2 training losses from metrics.json files into PNG charts."""

from __future__ import annotations

import argparse
import json
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
SLATE = "#64748b"

SERIES_COLORS = [BLUE, TEAL, ORANGE, GREEN, RED, PURPLE, SLATE]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "METRICS_JSON"),
        help="One run label plus its Detectron2 metrics.json path. Repeat this flag.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/visualizations/training_losses",
        help="Directory for generated PNG plots.",
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
FONT_14 = load_font(14)
FONT_16 = load_font(16)
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


def load_metrics(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_series(rows, metric: str):
    points = []
    for row in rows:
        if "iteration" in row and metric in row:
            value = row[metric]
            if isinstance(value, (int, float)):
                points.append((int(row["iteration"]), float(value)))
    return points


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    import math

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
    return nice * magnitude


def new_canvas(width: int, height: int):
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    return image, draw


def draw_axes(draw, x0, y0, plot_w, plot_h, x_max, y_max, y_label):
    draw.line((x0, y0 - plot_h, x0, y0), fill=TEXT, width=2)
    draw.line((x0, y0, x0 + plot_w, y0), fill=TEXT, width=2)

    for i in range(6):
        value = y_max * i / 5
        y = y0 - plot_h * i / 5
        draw.line((x0, y, x0 + plot_w, y), fill=GRID, width=1)
        draw_right(draw, x0 - 10, y, f"{value:.2f}", FONT_11, MUTED)

    for i in range(6):
        value = x_max * i / 5
        x = x0 + plot_w * i / 5
        draw.line((x, y0, x, y0 + 4), fill=TEXT, width=1)
        draw_centered(draw, x, y0 + 20, str(int(round(value))), FONT_11, MUTED)

    draw.text((26, y0 - plot_h / 2), y_label, font=FONT_12, fill=MUTED)
    draw_centered(draw, x0 + plot_w / 2, y0 + 46, "Iteration", FONT_12, MUTED)


def draw_legend(draw, x: int, y: int, items):
    for idx, (label, color) in enumerate(items):
        yi = y + idx * 22
        draw.rounded_rectangle((x, yi - 9, x + 14, yi + 5), radius=3, fill=color)
        draw.text((x + 22, yi - 12), label, font=FONT_12, fill=MUTED)


def plot_metric(runs, metric: str, title: str, out_path: Path):
    width, height = 1280, 760
    x0, y0 = 110, 650
    plot_w, plot_h = 980, 500

    run_points = []
    x_max = 1
    y_max_raw = 1.0
    for run in runs:
        points = extract_series(run["rows"], metric)
        if points:
            x_max = max(x_max, max(x for x, _ in points))
            y_max_raw = max(y_max_raw, max(y for _, y in points))
        run_points.append(points)

    y_max = nice_max(y_max_raw * 1.10)

    image, draw = new_canvas(width, height)
    draw_centered(draw, width / 2, 40, title, FONT_28, TEXT)
    draw_centered(draw, width / 2, 72, f"Metric: {metric}", FONT_13, MUTED)
    draw_axes(draw, x0, y0, plot_w, plot_h, x_max, y_max, "Loss")

    legend_items = []
    for idx, run in enumerate(runs):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        legend_items.append((run["label"], color))
        points = run_points[idx]
        if len(points) < 2:
            continue
        scaled = []
        for x, y in points:
            px = x0 + plot_w * x / x_max
            py = y0 - plot_h * y / y_max
            scaled.append((px, py))
        for p1, p2 in zip(scaled, scaled[1:]):
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=3)

        last_x, last_y = scaled[-1]
        draw.ellipse((last_x - 4, last_y - 4, last_x + 4, last_y + 4), fill=color)
        last_value = points[-1][1]
        draw.text((last_x + 8, last_y - 10), f"{last_value:.3f}", font=FONT_11, fill=color)

    draw_legend(draw, 1120, 120, legend_items)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")


def build_dashboard(runs, out_dir: Path):
    width, height = 1280, 760
    image, draw = new_canvas(width, height)
    draw_centered(draw, width / 2, 42, "Training Loss Summary", FONT_28, TEXT)
    draw_centered(draw, width / 2, 74, "Quick comparison across detector runs", FONT_13, MUTED)

    cards = [
        ("Runs loaded", str(len(runs)), BLUE, 48, 120),
        ("Max iteration", str(max(max((row.get("iteration", 0) for row in run["rows"]), default=0) for run in runs)), TEAL, 346, 120),
        ("Suggested main plot", "total_loss", ORANGE, 644, 120),
        ("Suggested detail plot", "loss_mask", GREEN, 942, 120),
    ]
    for title, value, color, x, y in cards:
        draw.rounded_rectangle((x, y, x + 250, y + 140), radius=18, fill=CARD, outline="#e7edf4")
        draw.text((x + 22, y + 18), title, font=FONT_18, fill=TEXT)
        draw.text((x + 22, y + 68), value, font=FONT_28, fill=color)

    draw.rounded_rectangle((48, 300, 1232, 680), radius=18, fill=CARD, outline="#e7edf4")
    draw.text((74, 326), "Loaded runs", font=FONT_18, fill=TEXT)
    y = 370
    for idx, run in enumerate(runs):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        draw.rounded_rectangle((76, y - 4, 90, y + 10), radius=3, fill=color)
        total = len(extract_series(run["rows"], "total_loss"))
        mask = len(extract_series(run["rows"], "loss_mask"))
        draw.text((104, y - 10), f"{run['label']}  |  total_loss points: {total}  |  loss_mask points: {mask}", font=FONT_13, fill=MUTED)
        y += 28

    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / "loss_dashboard.png", format="PNG")


def main():
    args = parse_args()
    if not args.run:
        raise SystemExit("Provide at least one --run LABEL METRICS_JSON pair.")

    runs = []
    for label, metrics_path in args.run:
        path = Path(metrics_path)
        runs.append({"label": label, "rows": load_metrics(path), "path": path})

    out_dir = Path(args.output_dir)
    build_dashboard(runs, out_dir)
    plot_metric(runs, "total_loss", "Detector training loss progression", out_dir / "loss_total.png")
    plot_metric(runs, "loss_mask", "Mask loss progression", out_dir / "loss_mask.png")
    plot_metric(runs, "loss_rpn_cls", "RPN classification loss progression", out_dir / "loss_rpn_cls.png")
    plot_metric(runs, "loss_box_reg_stage0", "Stage-0 box regression loss progression", out_dir / "loss_box_reg_stage0.png")

    print(f"Saved loss plots to {out_dir}")


if __name__ == "__main__":
    main()
