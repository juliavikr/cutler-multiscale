#!/usr/bin/env python3
"""
Reproducible end-to-end pipeline for the final 5-class TinyImageNet study.

The pipeline:
1. downloads required data (TinyImageNet subset source, DINO weights, COCO val)
2. prepares the locked 5-class subset
3. generates baseline single-scale MaskCut pseudo-labels
4. generates the latest refined hybrid multi-scale pseudo-labels
5. merges baseline + hybrid crop-only masks
6. trains one or more detector variants
7. evaluates the trained checkpoints on class-agnostic COCO

It intentionally avoids Slurm-specific logic so the same entrypoint can be run
inside an activated environment on the cluster or on another compatible machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_RUN_ROOT = REPO_ROOT / "experiments" / "repro_runs"

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DINO_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

CLASSES_5 = [
    "n01443537",
    "n02123045",
    "n02281406",
    "n02410509",
    "n02906734",
]

TRAINING_VARIANTS = ("baseline", "hybrid", "combined")
STEPS = ("download", "prepare", "baseline", "hybrid", "merge", "train", "eval")
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl")


def parse_csv_list(text: str, allowed: Iterable[str] | None = None) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if allowed is None:
        return values
    allowed_set = set(allowed)
    invalid = [item for item in values if item not in allowed_set]
    if invalid:
        raise SystemExit(f"Unknown values {invalid}; allowed values are: {sorted(allowed_set)}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out_handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_handle.write(chunk)
    tmp_path.replace(destination)


def ensure_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    download_file(url, destination)


def extract_zip_once(zip_path: Path, target_dir: Path, expected_path: Path) -> None:
    if expected_path.exists():
        return
    print(f"Extracting {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def ensure_tiny_imagenet(data_root: Path) -> Path:
    archive_dir = data_root / "tiny-imagenet"
    zip_path = archive_dir / "tiny-imagenet-200.zip"
    dataset_dir = data_root / "tiny-imagenet-200"
    ensure_file(TINY_IMAGENET_URL, zip_path)
    extract_zip_once(zip_path, data_root, dataset_dir / "train")
    return dataset_dir


def ensure_coco_eval_data(data_root: Path) -> None:
    coco_root = data_root / "coco"
    val_zip = coco_root / "val2017.zip"
    ann_zip = coco_root / "annotations_trainval2017.zip"
    ensure_file(COCO_VAL_URL, val_zip)
    ensure_file(COCO_ANN_URL, ann_zip)
    extract_zip_once(val_zip, coco_root, coco_root / "val2017")
    extract_zip_once(ann_zip, coco_root, coco_root / "annotations" / "instances_val2017.json")

    cls_agnostic = coco_root / "annotations" / "coco_cls_agnostic_instances_val2017.json"
    if not cls_agnostic.exists():
        source = coco_root / "annotations" / "instances_val2017.json"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "make_cls_agnostic_coco.py"),
                "--input",
                str(source),
                "--output",
                str(cls_agnostic),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )


def ensure_dino_weights(data_root: Path) -> Path:
    weights_path = data_root / "weights" / "dino_deitsmall8_pretrain.pth"
    ensure_file(DINO_URL, weights_path)
    return weights_path


def copy_tree_if_missing(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(source, destination)


def prepare_tinyimagenet_5_subset(data_root: Path, classes: Sequence[str]) -> Tuple[Path, Path]:
    source_root = data_root / "tiny-imagenet-200" / "train"
    subset_root = data_root / "tiny-imagenet-5"
    train_root = subset_root / "train"
    flat_root = subset_root / "train_flat"
    annotations_root = subset_root / "annotations"
    train_root.mkdir(parents=True, exist_ok=True)
    flat_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    for class_id in classes:
        source_class = source_root / class_id
        if not source_class.exists():
            raise FileNotFoundError(f"Missing TinyImageNet class folder: {source_class}")

        target_class = train_root / class_id
        copy_tree_if_missing(source_class, target_class)

        flat_class = flat_root / class_id
        flat_class.mkdir(parents=True, exist_ok=True)
        source_images = sorted((source_class / "images").glob("*"))
        for image_path in source_images:
            target_image = flat_class / image_path.name
            if not target_image.exists():
                shutil.copy2(image_path, target_image)

    class_manifest = {
        "classes": list(classes),
        "num_classes": len(classes),
        "images_per_class": {
            class_id: len(list((flat_root / class_id).glob("*")))
            for class_id in classes
        },
    }
    (annotations_root / "subset_manifest.json").write_text(json.dumps(class_manifest, indent=2), encoding="utf-8")
    return flat_root, annotations_root


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    print("")
    print(f"=== Running: {' '.join(command)}")
    print(f"    cwd: {cwd}")
    print(f"    log: {log_path}")
    print("")

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def locate_single_json(out_dir: Path, *, fixed_size: int, tau: float, n_masks: int) -> Path:
    pattern = f"imagenet_train_fixsize{fixed_size}_tau{tau}_N{n_masks}*.json"
    candidates = sorted(
        path
        for path in out_dir.glob(pattern)
        if not path.name.endswith(("_normal.json", "_raw_multiscale.json", "_multiscale.json", "_combined.json", "_candidate_debug.json"))
        and path.name != "checkpoint.json"
    )
    if not candidates:
        raise FileNotFoundError(f"Could not locate primary JSON in {out_dir} with pattern {pattern}")
    return candidates[-1]


def locate_split_json(out_dir: Path, split_name: str) -> Path:
    candidates = sorted(path for path in out_dir.glob(f"*_${split_name}.json"))
    if not candidates:
        candidates = sorted(path for path in out_dir.glob(f"*_{split_name}.json"))
    if not candidates:
        raise FileNotFoundError(f"Could not locate split JSON '*_{split_name}.json' in {out_dir}")
    return candidates[-1]


def copy_with_hash(source: Path, destination: Path) -> Dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return describe_artifact(destination)


def describe_artifact(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": sha256_file(path),
    }


def parse_eval_metrics(log_path: Path) -> Dict[str, Dict[str, float]]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    results: Dict[str, Dict[str, float]] = {}
    current_task = None
    for index, line in enumerate(lines):
        if "copypaste: Task:" in line:
            current_task = line.split("Task:", 1)[1].strip().lower()
            continue
        if current_task is None or "copypaste:" not in line:
            continue
        payload = line.split("copypaste:", 1)[1].strip()
        if payload.startswith("AP,AP50,AP75,APs,APm,APl"):
            if index + 1 >= len(lines):
                continue
            values_line = lines[index + 1]
            if "copypaste:" not in values_line:
                continue
            values_text = values_line.split("copypaste:", 1)[1].strip()
            try:
                values = [float(item) for item in values_text.split(",")]
            except ValueError:
                continue
            if len(values) != len(METRICS):
                continue
            results[current_task] = dict(zip(METRICS, values))
            current_task = None
    return results


def write_summary(
    run_root: Path,
    manifest: Dict[str, object],
    pseudo_stats: Dict[str, Dict[str, float]],
    eval_results: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    summary_json = run_root / "summary.json"
    summary_md = run_root / "summary.md"
    summary_csv = run_root / "summary.csv"

    payload = {
        "manifest": manifest,
        "pseudo_label_stats": pseudo_stats,
        "evaluation": eval_results,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Latest Pipeline Summary",
        "",
        "## Pseudo-label stats",
        "",
        "| Variant | Images | Annotations | Avg masks / image |",
        "|---|---:|---:|---:|",
    ]
    for variant in ("baseline", "hybrid", "combined"):
        if variant not in pseudo_stats:
            continue
        stats = pseudo_stats[variant]
        lines.append(
            f"| {variant} | {int(stats['images'])} | {int(stats['annotations'])} | {stats['avg_masks_per_image']:.3f} |"
        )

    if eval_results:
        lines.extend(
            [
                "",
                "## Detector evaluation",
                "",
                "| Variant | BBOX AP | AP50 | AP75 | APs | APm | APl | SEGM AP | SEGM AP50 | SEGM AP75 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant, tasks in eval_results.items():
            bbox = tasks.get("bbox", {})
            segm = tasks.get("segm", {})
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    variant,
                    bbox.get("AP", float("nan")),
                    bbox.get("AP50", float("nan")),
                    bbox.get("AP75", float("nan")),
                    bbox.get("APs", float("nan")),
                    bbox.get("APm", float("nan")),
                    bbox.get("APl", float("nan")),
                    segm.get("AP", float("nan")),
                    segm.get("AP50", float("nan")),
                    segm.get("AP75", float("nan")),
                )
            )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "bbox_AP",
                "bbox_AP50",
                "bbox_AP75",
                "bbox_APs",
                "bbox_APm",
                "bbox_APl",
                "segm_AP",
                "segm_AP50",
                "segm_AP75",
                "images",
                "annotations",
                "avg_masks_per_image",
            ]
        )
        all_variants = sorted(set(pseudo_stats) | set(eval_results))
        for variant in all_variants:
            bbox = eval_results.get(variant, {}).get("bbox", {})
            segm = eval_results.get(variant, {}).get("segm", {})
            stats = pseudo_stats.get(variant, {})
            writer.writerow(
                [
                    variant,
                    bbox.get("AP", ""),
                    bbox.get("AP50", ""),
                    bbox.get("AP75", ""),
                    bbox.get("APs", ""),
                    bbox.get("APm", ""),
                    bbox.get("APl", ""),
                    segm.get("AP", ""),
                    segm.get("AP50", ""),
                    segm.get("AP75", ""),
                    stats.get("images", ""),
                    stats.get("annotations", ""),
                    stats.get("avg_masks_per_image", ""),
                ]
            )


def compute_pseudo_stats(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = len(data.get("images", []))
    annotations = len(data.get("annotations", []))
    avg_masks = annotations / images if images else 0.0
    return {
        "images": images,
        "annotations": annotations,
        "avg_masks_per_image": avg_masks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default="latest_hybrid_pipeline")
    parser.add_argument("--steps", default="all", help="comma-separated subset of steps or 'all'")
    parser.add_argument("--variants", default="combined", help="comma-separated training/eval variants: baseline,hybrid,combined")
    parser.add_argument("--seed", type=int, default=42, help="Detectron2 seed for reproducible reruns (-1 keeps upstream default)")
    parser.add_argument("--force-download", action="store_true", help="re-download archives and weights even if they exist")
    args = parser.parse_args()

    selected_steps = list(STEPS) if args.steps == "all" else parse_csv_list(args.steps, STEPS)
    selected_variants = parse_csv_list(args.variants, TRAINING_VARIANTS)

    data_root = args.data_root.resolve()
    run_root = (args.run_root / args.run_name).resolve()
    logs_root = run_root / "logs"
    pseudo_root = run_root / "pseudo_labels"
    train_root = run_root / "training"
    eval_root = run_root / "eval"

    run_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    pseudo_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    if args.force_download:
        for path in (
            data_root / "tiny-imagenet" / "tiny-imagenet-200.zip",
            data_root / "coco" / "val2017.zip",
            data_root / "coco" / "annotations_trainval2017.zip",
            data_root / "weights" / "dino_deitsmall8_pretrain.pth",
        ):
            if path.exists():
                path.unlink()

    manifest: Dict[str, object] = {
        "run_name": args.run_name,
        "run_root": str(run_root),
        "data_root": str(data_root),
        "steps": selected_steps,
        "variants": selected_variants,
        "seed": args.seed,
        "classes_5": CLASSES_5,
        "artifacts": {},
        "started_at": time.time(),
    }

    if "download" in selected_steps:
        ensure_tiny_imagenet(data_root)
        ensure_coco_eval_data(data_root)
        dino_weights = ensure_dino_weights(data_root)
        manifest["artifacts"]["dino_weights"] = describe_artifact(dino_weights)

    if "prepare" in selected_steps:
        train_flat_root, annotations_root = prepare_tinyimagenet_5_subset(data_root, CLASSES_5)
    else:
        train_flat_root = data_root / "tiny-imagenet-5" / "train_flat"
        annotations_root = data_root / "tiny-imagenet-5" / "annotations"
    manifest["artifacts"]["train_flat_root"] = str(train_flat_root)
    manifest["artifacts"]["annotations_root"] = str(annotations_root)

    dino_weights = data_root / "weights" / "dino_deitsmall8_pretrain.pth"
    class_count = len(CLASSES_5)

    if "baseline" in selected_steps:
        baseline_out = run_root / "generation" / "baseline"
        baseline_out.mkdir(parents=True, exist_ok=True)
        baseline_log = logs_root / "baseline_maskcut.log"
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "multiscale" / "multiscale_maskcut.py"),
                "--out-dir",
                str(baseline_out),
                "--dataset-path",
                str(train_flat_root),
                "--vit-arch",
                "small",
                "--vit-feat",
                "k",
                "--patch-size",
                "8",
                "--tau",
                "0.15",
                "--fixed_size",
                "480",
                "--N",
                "3",
                "--num-folder-per-job",
                str(class_count),
                "--job-index",
                "0",
                "--pretrain_path",
                str(dino_weights),
            ],
            cwd=REPO_ROOT,
            log_path=baseline_log,
        )
        baseline_primary = locate_single_json(baseline_out, fixed_size=480, tau=0.15, n_masks=3)
        manifest["artifacts"]["baseline_generation"] = copy_with_hash(baseline_primary, pseudo_root / "baseline.json")

    if "hybrid" in selected_steps:
        hybrid_out = run_root / "generation" / "hybrid_best"
        hybrid_out.mkdir(parents=True, exist_ok=True)
        hybrid_log = logs_root / "hybrid_maskcut.log"
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "multiscale" / "multiscale_maskcut.py"),
                "--out-dir",
                str(hybrid_out),
                "--dataset-path",
                str(train_flat_root),
                "--vit-arch",
                "small",
                "--vit-feat",
                "k",
                "--patch-size",
                "8",
                "--tau",
                "0.2",
                "--fixed_size",
                "480",
                "--N",
                "1",
                "--num-folder-per-job",
                str(class_count),
                "--job-index",
                "0",
                "--pretrain_path",
                str(dino_weights),
                "--multi-crop",
                "--ms-preset",
                "small",
                "--primary-output",
                "multiscale",
                "--log-every",
                "50",
            ],
            cwd=REPO_ROOT,
            log_path=hybrid_log,
        )
        hybrid_multiscale = locate_split_json(hybrid_out, "multiscale")
        manifest["artifacts"]["hybrid_generation"] = copy_with_hash(hybrid_multiscale, pseudo_root / "hybrid_multiscale.json")

    if "merge" in selected_steps:
        combine_log = logs_root / "combine_masks.log"
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "combine_pseudo_labels.py"),
                "--baseline-json",
                str(pseudo_root / "baseline.json"),
                "--multiscale-json",
                str(pseudo_root / "hybrid_multiscale.json"),
                "--output-json",
                str(pseudo_root / "combined.json"),
                "--sort-multiscale-by-area",
            ],
            cwd=REPO_ROOT,
            log_path=combine_log,
        )
        manifest["artifacts"]["combined_generation"] = describe_artifact(pseudo_root / "combined.json")

    pseudo_label_paths = {
        "baseline": pseudo_root / "baseline.json",
        "hybrid": pseudo_root / "hybrid_multiscale.json",
        "combined": pseudo_root / "combined.json",
    }
    pseudo_stats = {
        variant: compute_pseudo_stats(path)
        for variant, path in pseudo_label_paths.items()
        if path.exists()
    }

    eval_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    config_path = REPO_ROOT / "CutLER" / "cutler" / "model_zoo" / "configs" / "CutLER-ImageNet" / "cascade_mask_rcnn_R_50_FPN.yaml"

    if "train" in selected_steps:
        for variant in selected_variants:
            json_path = pseudo_label_paths[variant]
            dataset_name = f"tinyimagenet_5c_repro_{variant}"
            train_dir = train_root / variant
            train_dir.mkdir(parents=True, exist_ok=True)
            train_log = logs_root / f"train_{variant}.log"
            command = [
                sys.executable,
                str(REPO_ROOT / "tools" / "train_wrapper_dynamic.py"),
                "--dataset-name",
                dataset_name,
                "--json-path",
                str(json_path),
                "--image-root",
                str(train_flat_root),
                "--num-gpus",
                "1",
                "--config-file",
                str(config_path),
                "DATASETS.TRAIN",
                f"(\"{dataset_name}\",)",
                "SOLVER.IMS_PER_BATCH",
                "8",
                "SOLVER.BASE_LR",
                "0.005",
                "SOLVER.MAX_ITER",
                "20000",
                "SOLVER.STEPS",
                "(15000,)",
                "SOLVER.WARMUP_ITERS",
                "1000",
                "DATALOADER.NUM_WORKERS",
                "2",
                "OUTPUT_DIR",
                str(train_dir),
            ]
            if args.seed >= 0:
                command.extend(["SEED", str(args.seed)])
            run_command(
                command,
                cwd=REPO_ROOT / "CutLER" / "cutler",
                log_path=train_log,
                env={
                    "DATA_ROOT": str(data_root),
                    "DETECTRON2_DATASETS": str(data_root),
                },
            )
            manifest["artifacts"][f"training_{variant}"] = {
                "dir": str(train_dir),
                "checkpoint": str(train_dir / "model_final.pth"),
            }

    if "eval" in selected_steps:
        ensure_coco_eval_data(data_root)
        for variant in selected_variants:
            checkpoint = train_root / variant / "model_final.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Expected checkpoint for {variant} at {checkpoint}")
            variant_eval_dir = eval_root / variant
            variant_eval_dir.mkdir(parents=True, exist_ok=True)
            eval_log = logs_root / f"eval_{variant}.log"
            run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "CutLER" / "cutler" / "train_net.py"),
                    "--num-gpus",
                    "1",
                    "--eval-only",
                    "--config-file",
                    str(config_path),
                    "DATASETS.TEST",
                    "(\"cls_agnostic_coco\",)",
                    "MODEL.WEIGHTS",
                    str(checkpoint),
                    "TEST.DETECTIONS_PER_IMAGE",
                    "100",
                    "OUTPUT_DIR",
                    str(variant_eval_dir),
                ],
                cwd=REPO_ROOT / "CutLER" / "cutler",
                log_path=eval_log,
                env={
                    "DATA_ROOT": str(data_root),
                    "DETECTRON2_DATASETS": str(data_root),
                },
            )
            eval_results[variant] = parse_eval_metrics(eval_log)
            manifest["artifacts"][f"eval_{variant}"] = {
                "dir": str(variant_eval_dir),
                "log": str(eval_log),
            }

    manifest["completed_at"] = time.time()
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(run_root, manifest, pseudo_stats, eval_results)
    print("")
    print(f"Pipeline finished. Summary: {run_root / 'summary.md'}")


if __name__ == "__main__":
    main()
