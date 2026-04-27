#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Dict, List


METRICS = ["AP", "AP50", "AP75", "APs", "APm", "APl"]


def parse_metrics_from_log(log_path: str) -> Dict[str, float]:
    metrics = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    lines = text.splitlines()
    for line in lines:
        line_clean = line.strip()
        for m in METRICS:
            pattern = rf"\b{re.escape(m)}\b[^0-9\-]*([0-9]+(?:\.[0-9]+)?)"
            hit = re.search(pattern, line_clean)
            if hit:
                metrics[m] = float(hit.group(1))

    # fallback to the common detectron2 "copypaste" block
    for i, line in enumerate(lines):
        if "copypaste:" not in line:
            continue
        if i + 2 >= len(lines):
            continue
        header = lines[i + 1].strip().split(",")
        values = lines[i + 2].strip().split(",")
        if len(header) != len(values):
            continue
        for h, v in zip(header, values):
            h = h.strip()
            v = v.strip()
            if h in METRICS:
                try:
                    metrics[h] = float(v)
                except ValueError:
                    pass

    return metrics


def find_eval_logs(root: str) -> List[str]:
    logs = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == "eval.log":
                logs.append(os.path.join(dirpath, name))
    return sorted(logs)


def main():
    parser = argparse.ArgumentParser("Rank runs by APs from eval logs")
    parser.add_argument("--root", type=str, default="output", help="directory to scan for eval.log files")
    parser.add_argument("--csv-out", type=str, default="", help="optional CSV output path")
    args = parser.parse_args()

    logs = find_eval_logs(args.root)
    rows = []
    for log_path in logs:
        metrics = parse_metrics_from_log(log_path)
        if "APs" not in metrics:
            continue
        row = {"run": os.path.relpath(os.path.dirname(log_path), args.root), "log": log_path}
        for m in METRICS:
            row[m] = metrics.get(m, float("nan"))
        rows.append(row)

    rows.sort(key=lambda r: r["APs"], reverse=True)

    if not rows:
        print(f"No eval logs with APs found under: {args.root}")
        return

    print("Ranked by APs (desc):")
    for idx, r in enumerate(rows, 1):
        print(
            f"{idx:>2}. {r['run']} | APs={r['APs']:.3f} | AP={r['AP']:.3f} "
            f"| AP50={r['AP50']:.3f} | AP75={r['AP75']:.3f} | APm={r['APm']:.3f} | APl={r['APl']:.3f}"
        )

    if args.csv_out:
        fieldnames = ["run", "log"] + METRICS
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nSaved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
