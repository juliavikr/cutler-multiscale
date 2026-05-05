"""Compare descriptive statistics between two COCO-format pseudo-label JSONs."""

import argparse
import json
import statistics
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


SMALL_MAX = 32 * 32       # 1024
MEDIUM_MAX = 96 * 96      # 9216


def compute_stats(data: dict) -> dict:
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    total_images = len(images)
    total_anns = len(annotations)

    # masks per image
    masks_per_image: dict[int, int] = {img["id"]: 0 for img in images}
    areas: list[float] = []

    for ann in annotations:
        img_id = ann["image_id"]
        masks_per_image[img_id] = masks_per_image.get(img_id, 0) + 1
        areas.append(float(ann.get("area", 0)))

    counts = list(masks_per_image.values())

    avg_masks = statistics.mean(counts) if counts else 0.0
    med_masks = statistics.median(counts) if counts else 0.0
    mean_area = statistics.mean(areas) if areas else 0.0
    med_area = statistics.median(areas) if areas else 0.0

    small = sum(1 for a in areas if a < SMALL_MAX)
    medium = sum(1 for a in areas if SMALL_MAX <= a < MEDIUM_MAX)
    large = sum(1 for a in areas if a >= MEDIUM_MAX)
    n = len(areas) or 1

    dist = {k: 0 for k in range(6)}  # 0,1,2,3,4,5+
    for c in counts:
        bucket = min(c, 5)
        dist[bucket] += 1

    return {
        "total_images": total_images,
        "total_anns": total_anns,
        "avg_masks": avg_masks,
        "med_masks": med_masks,
        "mean_area": mean_area,
        "med_area": med_area,
        "small_count": small,
        "small_pct": 100 * small / n,
        "medium_count": medium,
        "medium_pct": 100 * medium / n,
        "large_count": large,
        "large_pct": 100 * large / n,
        "dist": dist,
    }


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def build_table(baseline: dict, hybrid: dict) -> str:
    rows = []

    def row(label, bval, hval):
        rows.append(f"| {label} | {fmt(bval)} | {fmt(hval)} |")

    header = (
        "| Metric | Baseline | Hybrid |\n"
        "|--------|----------|--------|\n"
    )

    row("Total images", baseline["total_images"], hybrid["total_images"])
    row("Total annotations (masks)", baseline["total_anns"], hybrid["total_anns"])
    row("Avg masks / image", baseline["avg_masks"], hybrid["avg_masks"])
    row("Median masks / image", baseline["med_masks"], hybrid["med_masks"])
    row("Mean mask area (px²)", baseline["mean_area"], hybrid["mean_area"])
    row("Median mask area (px²)", baseline["med_area"], hybrid["med_area"])

    rows.append("|--------|----------|--------|")
    rows.append("| **Size bin** | | |")

    def size_row(label, bcount, bpct, hcount, hpct):
        rows.append(
            f"| {label} | {bcount:,} ({bpct:.1f}%) | {hcount:,} ({hpct:.1f}%) |"
        )

    size_row(
        "Small (<32² px)",
        baseline["small_count"], baseline["small_pct"],
        hybrid["small_count"], hybrid["small_pct"],
    )
    size_row(
        "Medium (32²–96² px)",
        baseline["medium_count"], baseline["medium_pct"],
        hybrid["medium_count"], hybrid["medium_pct"],
    )
    size_row(
        "Large (>96² px)",
        baseline["large_count"], baseline["large_pct"],
        hybrid["large_count"], hybrid["large_pct"],
    )

    rows.append("|--------|----------|--------|")
    rows.append("| **Masks-per-image distribution** | | |")

    for k in range(5):
        label = f"Images with {k} masks"
        rows.append(
            f"| {label} | {baseline['dist'][k]:,} | {hybrid['dist'][k]:,} |"
        )
    rows.append(
        f"| Images with 5+ masks | {baseline['dist'][5]:,} | {hybrid['dist'][5]:,} |"
    )

    return header + "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--hybrid-json", required=True)
    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline_json}")
    baseline_data = load_json(args.baseline_json)
    print(f"Loading hybrid:   {args.hybrid_json}")
    hybrid_data = load_json(args.hybrid_json)

    baseline_stats = compute_stats(baseline_data)
    hybrid_stats = compute_stats(hybrid_data)

    table = build_table(baseline_stats, hybrid_stats)

    out_path = Path(__file__).parent.parent / "presentation" / "01_results" / "pseudo_label_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_md = (
        "# Pseudo-label Statistics: Baseline vs Hybrid\n\n"
        f"- Baseline: `{args.baseline_json}`\n"
        f"- Hybrid: `{args.hybrid_json}`\n\n"
    )

    full_md = header_md + table
    out_path.write_text(full_md)
    print(f"\nSaved to {out_path}\n")
    print(full_md)


if __name__ == "__main__":
    main()
