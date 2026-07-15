"""Build the evidence package required by the FDSA-YOLO JRS manuscript.

This script is intentionally independent from ``run_paper_experiments.py``. It
re-runs only the two missing paper-protocol seeds and derives post-training
evidence from the three core models:

* YOLOv8n
* SCFR+PFM
* FDSA-YOLO

The public phases are ``dry``, ``seed``, ``size``, ``mechanism``,
``qualitative``, and ``all``. Training jobs use one A10 each; latency is always
measured serially after the parallel workers have exited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO, __version__ as ultralytics_version


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "vis.yaml"
DEFAULT_VIS_ROOT = ROOT / "datasets"
DEFAULT_PAPER_ROOT = ROOT / "runs" / "paper"
DEFAULT_OUTPUT = ROOT / "runs" / "jrs_evidence"

IMGSZ = 640
EPOCHS = 150
PATIENCE = 20
TRAIN_BATCH = 8
VAL_BATCH = 56
WORKERS = 2
SEEDS_TO_TRAIN = (1, 2)
TRAIN_STATIC_KWARGS = {
    "save": True,
    "save_period": -1,
    "cache": False,
    "pretrained": True,
    "optimizer": "auto",
    "deterministic": True,
    "rect": False,
    "cos_lr": False,
    "close_mosaic": 10,
    "amp": True,
    "half": False,
    "fraction": 1.0,
    "freeze": None,
    "multi_scale": 0.0,
    "dropout": 0.0,
    "val": True,
    "plots": True,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "nbs": 64,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
}
SIZE_BINS = (
    ("tiny", 0.0, 0.00025),
    ("small", 0.00025, 0.001),
    ("medium", 0.001, 0.005),
    ("large", 0.005, 1.0e10),
)
CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
SOURCE_NAMES = ("P4-up", "P3-skip", "P4-fused", "P5-SPPF")


@dataclass(frozen=True)
class CoreModel:
    key: str
    label: str
    cfg_candidates: tuple[str, ...]
    seed0_run: str

    def resolve_cfg(self) -> Path:
        for rel in self.cfg_candidates:
            path = ROOT / rel
            if path.exists():
                return path
        raise FileNotFoundError(f"No model YAML found for {self.label}: {self.cfg_candidates}")

    def seed_run(self, seed: int) -> str:
        return f"jrs-{self.key}-seed{seed}"


CORE_MODELS = (
    CoreModel(
        "yolov8n",
        "YOLOv8n",
        ("models/yolov8n.yaml", "models/yolov8.yaml"),
        "paper-yolov8n-seed0",
    ),
    CoreModel(
        "scfr-pfm",
        "SCFR+PFM",
        ("models/yolov8n_scfr_pfm.yaml",),
        "paper-a1-n-seed0",
    ),
    CoreModel(
        "fdsa-yolo",
        "FDSA-YOLO",
        ("models/yolov8n_fdsa.yaml",),
        "paper-fdsa-yolo-n-seed0",
    ),
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: jsonable(v) for k, v in row.items()} for row in rows])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gpu_list(raw: str) -> list[str]:
    gpus = [part.strip() for part in raw.split(",") if part.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required.")
    return gpus


def model_by_key(key: str) -> CoreModel:
    for spec in CORE_MODELS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def seed0_weights(spec: CoreModel, args: argparse.Namespace) -> Path:
    return Path(args.paper_root).resolve() / "detect" / spec.seed0_run / "weights" / "best.pt"


def seed0_args(spec: CoreModel, args: argparse.Namespace) -> Path:
    return Path(args.paper_root).resolve() / "detect" / spec.seed0_run / "args.yaml"


def seed_weights(spec: CoreModel, seed: int, args: argparse.Namespace) -> Path:
    if seed == 0:
        return seed0_weights(spec, args)
    return Path(args.output).resolve() / "detect" / spec.seed_run(seed) / "weights" / "best.pt"


def image_and_label_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.vis_root).resolve()
    return root / "VisDrone2019-DET-val" / "images", root / "VisDrone2019-DET-val" / "labels"


def environment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    files = [Path(__file__).resolve(), ROOT / "ultralytics/nn/modules/block.py", ROOT / "ultralytics/nn/tasks.py"]
    files.extend(spec.resolve_cfg() for spec in CORE_MODELS)
    images, labels = image_and_label_dirs(args)
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ultralytics": ultralytics_version,
        "data": str(Path(args.data).resolve()),
        "val_images": len(list(images.glob("*"))) if images.exists() else None,
        "val_labels": len(list(labels.glob("*.txt"))) if labels.exists() else None,
        "files": [{"path": path.as_posix(), "sha256": sha256(path)} for path in files if path.exists()],
        "models": [asdict(spec) | {"resolved_cfg": spec.resolve_cfg().as_posix()} for spec in CORE_MODELS],
        "training": {
            "imgsz": IMGSZ,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch": TRAIN_BATCH,
            "device_scope": "one A10 per model (no multi-GPU DDP)",
            "model_initialization": "constructed from YAML; no checkpoint weights are transferred",
            "pretrained_argument": True,
            "optimizer_argument": "auto",
            "expected_optimizer": "MuSGD(lr=0.01, momentum=0.9)",
            "seeds": [0, 1, 2],
        },
        "validation": {
            "imgsz": IMGSZ,
            "batch": VAL_BATCH,
            "devices": args.val_devices,
            "half": False,
        },
        "size_bins": [{"name": n, "min_area_ratio": lo, "max_area_ratio": hi} for n, lo, hi in SIZE_BINS],
    }


def audit_seed0_protocol(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Verify the actual seed0 args, not the obsolete manifest-level planned batch."""
    expected = {
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch": TRAIN_BATCH,
        "imgsz": IMGSZ,
        "workers": WORKERS,
        **TRAIN_STATIC_KWARGS,
    }
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for spec in CORE_MODELS:
        path = seed0_args(spec, args)
        if not path.exists():
            failures.append(f"{spec.label}: missing {path}")
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        device = str(payload.get("device", ""))
        if "," in device:
            mismatches["device_scope"] = {"expected": "one GPU id", "actual": device}
        rows.append(
            {
                "model": spec.label,
                "args_yaml": path.as_posix(),
                "actual_device": device,
                "actual_batch": payload.get("batch"),
                "protocol_match": not mismatches,
                "mismatches": mismatches,
            }
        )
        if mismatches:
            failures.append(f"{spec.label}: {json.dumps(mismatches, ensure_ascii=False)}")
    write_json(Path(args.output).resolve() / "seed0_protocol_audit.json", rows)
    if failures:
        raise RuntimeError("Seed0 protocol audit failed:\n  " + "\n  ".join(failures))
    return rows


def dry_check(args: argparse.Namespace) -> None:
    rows = []
    images, labels = image_and_label_dirs(args)
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not images.exists() or not labels.exists():
        raise FileNotFoundError(f"VisDrone validation images/labels not found below {args.vis_root}")
    for spec in CORE_MODELS:
        cfg = spec.resolve_cfg()
        model = YOLO(cfg.as_posix())
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        rows.append(
            {
                "model": spec.label,
                "cfg": cfg.as_posix(),
                "params_m": round(params, 6),
                "seed0_weights": seed0_weights(spec, args).as_posix(),
                "seed0_exists": seed0_weights(spec, args).exists(),
            }
        )
        del model
    out = Path(args.output).resolve()
    write_csv(out / "dry_check.csv", rows)
    protocol_rows = audit_seed0_protocol(args)
    write_json(out / "environment_manifest.json", environment_manifest(args))
    print("\nDry check")
    for row in rows:
        print(f"  {row['model']}: params={row['params_m']:.6f}M, seed0={row['seed0_exists']}")
    print("  Seed0 protocol: single A10 per model, batch=8; all audited fields match.")
    for row in protocol_rows:
        print(f"    {row['model']}: GPU {row['actual_device']}, batch={row['actual_batch']}")
    missing = [row["model"] for row in rows if not row["seed0_exists"]]
    if missing:
        raise FileNotFoundError(
            "Missing paper seed0 weights for: " + ", ".join(missing) + ". Upload runs/paper/detect first."
        )


def train_worker(args: argparse.Namespace) -> None:
    key, seed_raw = args.job.rsplit(":", 1)
    seed = int(seed_raw)
    spec = model_by_key(key)
    output = Path(args.output).resolve()
    run_name = spec.seed_run(seed)
    run_dir = output / "detect" / run_name
    weights = run_dir / "weights" / "best.pt"
    if weights.exists() and not args.force:
        print(f"[skip] {run_name}: {weights}")
        return
    status = {
        "model": spec.label,
        "key": spec.key,
        "seed": seed,
        "device": args.device,
        "cfg": spec.resolve_cfg().as_posix(),
        "data": str(Path(args.data).resolve()),
        "batch": TRAIN_BATCH,
        "device_scope": "single GPU",
        "pretrained_argument": True,
        "model_initialization": "YAML random initialization; no checkpoint transfer",
        "optimizer": "auto (expected MuSGD)",
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        model = YOLO(spec.resolve_cfg().as_posix())
        model.train(
            data=str(Path(args.data).resolve()),
            imgsz=IMGSZ,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=TRAIN_BATCH,
            workers=WORKERS,
            device=args.device,
            seed=seed,
            project=(output / "detect").as_posix(),
            name=run_name,
            exist_ok=True,
            resume=False,
            **TRAIN_STATIC_KWARGS,
        )
        status["status"] = "trained"
    except BaseException as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "jrs_run_status.json", status)
        raise
    status["finished"] = datetime.now().isoformat(timespec="seconds")
    status["weights"] = weights.as_posix()
    write_json(run_dir / "jrs_run_status.json", status)


def launch_seed_workers(args: argparse.Namespace) -> None:
    gpus = parse_gpu_list(args.gpus)
    jobs = [(spec, seed) for spec in CORE_MODELS for seed in SEEDS_TO_TRAIN]
    if len(gpus) < len(jobs):
        raise ValueError(f"Seed phase needs {len(jobs)} GPU ids, got {len(gpus)}: {args.gpus}")
    output = Path(args.output).resolve()
    scheduler_dir = output / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, subprocess.Popen[Any], str, Path, Any]] = []
    print("\nParallel same-protocol seed schedule")
    for gpu, (spec, seed) in zip(gpus, jobs):
        job = f"{spec.key}:{seed}"
        cmd = [
            sys.executable,
            Path(__file__).resolve().as_posix(),
            "--phase",
            "seed-worker",
            "--job",
            job,
            "--device",
            gpu,
            "--data",
            str(Path(args.data).resolve()),
            "--vis-root",
            str(Path(args.vis_root).resolve()),
            "--paper-root",
            str(Path(args.paper_root).resolve()),
            "--output",
            str(Path(args.output).resolve()),
        ]
        if args.force:
            cmd.append("--force")
        print(f"  GPU {gpu}: {spec.label} seed{seed}")
        # Keep only stderr while a worker runs. Successful logs are removed; failures retain a short tail.
        log_path = scheduler_dir / f"worker_{job.replace(':', '_')}_gpu{gpu}.stderr.tmp"
        log_handle = log_path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(cmd, cwd=ROOT.as_posix(), stdout=subprocess.DEVNULL, stderr=log_handle)
        procs.append((gpu, proc, job, log_path, log_handle))
    failures = []
    for gpu, proc, job, log_path, log_handle in procs:
        code = proc.wait()
        log_handle.close()
        if code:
            failure_path = scheduler_dir / f"worker_{job.replace(':', '_')}_gpu{gpu}_failure_tail.log"
            if log_path.exists():
                with log_path.open("rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 65536))
                    tail = f.read().decode("utf-8", errors="replace")
                failure_path.write_text(tail or "No stderr captured.", encoding="utf-8")
                log_path.unlink()
            else:
                failure_path.write_text("No stderr log was created.", encoding="utf-8")
            failures.append(f"{job}@GPU{gpu}(exit={code}; {failure_path.name})")
        elif log_path.exists():
            log_path.unlink()
    if failures:
        raise RuntimeError("Seed workers failed: " + ", ".join(failures))


def result_metrics(results: Any) -> dict[str, Any]:
    metrics = dict(getattr(results, "results_dict", {}) or {})
    speed = dict(getattr(results, "speed", {}) or {})
    return {str(k): jsonable(v) for k, v in metrics.items()} | {"ultralytics_speed_ms": speed}


def benchmark_latency(weights: Path, device: str, args: argparse.Namespace) -> dict[str, Any]:
    model = YOLO(weights.as_posix()).model.to(torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu"))
    model = model.float().eval()
    target_device = next(model.parameters()).device
    image = torch.randn(1, 3, IMGSZ, IMGSZ, device=target_device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(args.latency_warmup):
            model(image)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        times = []
        for _ in range(args.latency_iters):
            start = time.perf_counter()
            model(image)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            times.append((time.perf_counter() - start) * 1000.0)
    median = float(np.median(times))
    return {
        "device": str(target_device),
        "protocol": "single-device FP32 batch1 forward only",
        "imgsz": IMGSZ,
        "warmup": args.latency_warmup,
        "iterations": args.latency_iters,
        "median_ms": median,
        "p90_ms": float(np.percentile(times, 90)),
        "fps": 1000.0 / median,
    }


def validate_checkpoint(spec: CoreModel, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    weights = seed_weights(spec, seed, args)
    if not weights.exists():
        raise FileNotFoundError(weights)
    output = Path(args.output).resolve()
    run_name = spec.seed0_run if seed == 0 else spec.seed_run(seed)
    save_name = f"val_{run_name}"
    summary_path = output / "val" / save_name / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    model = YOLO(weights.as_posix())
    info = model.info(detailed=False, verbose=False, imgsz=IMGSZ)
    _, params, _, gflops = info if info else (None, None, None, None)
    results = model.val(
        data=str(Path(args.data).resolve()),
        imgsz=IMGSZ,
        batch=VAL_BATCH,
        workers=WORKERS,
        device=args.val_devices,
        half=False,
        plots=False,
        save_json=False,
        project=(output / "val").as_posix(),
        name=save_name,
        exist_ok=True,
    )
    summary = {
        "model_key": spec.key,
        "method": spec.label,
        "seed": seed,
        "weights": weights.as_posix(),
        "weights_sha256": sha256(weights),
        "cfg": spec.resolve_cfg().as_posix(),
        "params_m": float(params) / 1e6 if params else None,
        "gflops": gflops,
        "metrics": result_metrics(results),
        "protocol": {
            "imgsz": IMGSZ,
            "val_batch": VAL_BATCH,
            "device": args.val_devices,
            "half": False,
        },
    }
    write_json(summary_path, summary)
    return summary


def metric_value(summary: dict[str, Any], *names: str) -> float:
    metrics = summary.get("metrics", {})
    for name in names:
        if name in metrics:
            return float(metrics[name])
    raise KeyError(f"Metric not found: {names}; available={list(metrics)}")


def validate_and_summarize_seeds(args: argparse.Namespace) -> None:
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in CORE_MODELS:
        for seed in (0, 1, 2):
            summaries[(spec.key, seed)] = validate_checkpoint(spec, seed, args)
    latency = {}
    for spec in CORE_MODELS:
        latency[spec.key] = benchmark_latency(seed0_weights(spec, args), args.latency_device, args)
    rows = []
    by_key: dict[str, list[float]] = {}
    for spec in CORE_MODELS:
        values = [
            metric_value(summaries[(spec.key, seed)], "metrics/mAP50-95(B)", "map50_95") for seed in (0, 1, 2)
        ]
        by_key[spec.key] = values
        rows.append(
            {
                "method": spec.label,
                "seed0": values[0],
                "seed1": values[1],
                "seed2": values[2],
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
                "fps_seed0": latency[spec.key]["fps"],
                "latency_median_ms_seed0": latency[spec.key]["median_ms"],
                "latency_p90_ms_seed0": latency[spec.key]["p90_ms"],
            }
        )
    delta_rows = []
    for seed in (0, 1, 2):
        delta_rows.append(
            {
                "seed": seed,
                "fdsa_minus_yolov8n": by_key["fdsa-yolo"][seed] - by_key["yolov8n"][seed],
                "fdsa_minus_scfr_pfm": by_key["fdsa-yolo"][seed] - by_key["scfr-pfm"][seed],
            }
        )
    fdsa_parent = [row["fdsa_minus_scfr_pfm"] for row in delta_rows]
    fdsa_base = [row["fdsa_minus_yolov8n"] for row in delta_rows]
    acceptance = {
        "mean_delta_vs_parent": statistics.mean(fdsa_parent),
        "positive_seeds_vs_parent": sum(delta > 0 for delta in fdsa_parent),
        "mean_delta_vs_yolov8n": statistics.mean(fdsa_base),
    }
    acceptance["passed"] = (
        acceptance["mean_delta_vs_parent"] >= 0.0015
        and acceptance["positive_seeds_vs_parent"] >= 2
        and acceptance["mean_delta_vs_yolov8n"] >= 0.015
    )
    out = Path(args.output).resolve() / "stability"
    write_csv(out / "three_seed_summary.csv", rows)
    write_csv(out / "seedwise_deltas.csv", delta_rows)
    write_json(out / "acceptance.json", acceptance)
    write_json(out / "latency_seed0.json", latency)
    print("\nThree-seed acceptance:", json.dumps(acceptance, indent=2))


def seed_phase(args: argparse.Namespace) -> None:
    launch_seed_workers(args)
    validate_and_summarize_seeds(args)


def load_yolo_ground_truth(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, Path]]:
    images_dir, labels_dir = image_and_label_dirs(args)
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    images = []
    annotations = []
    id_to_path = {}
    ann_id = 1
    for image_id, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": image_path.name, "width": 1.0, "height": 1.0, "orig_width": width, "orig_height": height})
        id_to_path[image_id] = image_path
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, xc, yc, w, h = int(float(parts[0])), *map(float, parts[1:5])
            x = max(0.0, xc - w / 2)
            y = max(0.0, yc - h / 2)
            w = min(w, 1.0 - x)
            h = min(h, 1.0 - y)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return images, annotations, id_to_path


def predictions_path(spec: CoreModel, args: argparse.Namespace) -> Path:
    return Path(args.output).resolve() / "predictions" / f"{spec.key}_seed0_normalized.json"


def collect_predictions(spec: CoreModel, args: argparse.Namespace, id_to_path: dict[int, Path]) -> list[dict[str, Any]]:
    cache = predictions_path(spec, args)
    if cache.exists() and not args.force:
        return json.loads(cache.read_text(encoding="utf-8"))
    model = YOLO(seed0_weights(spec, args).as_posix())
    # Do not pass a Python list of paths here. This Ultralytics version treats lists as in-memory images and sends the
    # entire list as one batch, bypassing the requested batch=1. A directory source uses LoadImagesAndVideos instead.
    ordered_inputs = list(id_to_path.items())
    image_dir = ordered_inputs[0][1].parent
    if any(path.parent != image_dir for _, path in ordered_inputs):
        raise RuntimeError("Size evaluation expects all VisDrone validation images in one directory.")
    path_to_id = {path.resolve(): image_id for image_id, path in ordered_inputs}
    rows: list[dict[str, Any]] = []
    results = model.predict(
        source=image_dir.as_posix(),
        imgsz=IMGSZ,
        conf=args.pred_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=args.analysis_device,
        batch=1,
        half=False,
        stream=True,
        verbose=False,
        save=False,
    )
    result_count = 0
    seen_ids: set[int] = set()
    for result_count, result in enumerate(results, start=1):
        image_path = Path(result.path).resolve()
        image_id = path_to_id.get(image_path)
        if image_id is None:
            raise RuntimeError(f"Predictor returned an unexpected image path for {spec.label}: {image_path}")
        if image_id in seen_ids:
            raise RuntimeError(f"Predictor returned a duplicate image for {spec.label}: {image_path}")
        seen_ids.add(image_id)
        height, width = result.orig_shape
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, category in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = box.tolist()
            rows.append(
                {
                    "image_id": image_id,
                    "category_id": int(category) + 1,
                    "bbox": [x1 / width, y1 / height, max(0.0, x2 - x1) / width, max(0.0, y2 - y1) / height],
                    "score": float(score),
                }
            )
    if result_count != len(ordered_inputs) or len(seen_ids) != len(ordered_inputs):
        raise RuntimeError(
            f"Predictor returned {result_count} results for {len(ordered_inputs)} sources while evaluating {spec.label}."
        )
    write_json(cache, rows)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def coco_ap_by_size(
    images: list[dict[str, Any]], annotations: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Size evaluation requires pycocotools: python -m pip install pycocotools") from exc
    coco_gt = COCO()
    coco_gt.dataset = {
        # Newer pycocotools releases expect these standard top-level COCO fields in loadRes().
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions) if predictions else coco_gt.loadRes([])
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.params.areaRng = [[lo, hi] for _, lo, hi in SIZE_BINS]
    evaluator.params.areaRngLbl = [name for name, _, _ in SIZE_BINS]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDet
    rows = []
    for area_index, (name, lo, hi) in enumerate(SIZE_BINS):
        values = precision[:, :, :, area_index, -1]
        valid = values[values > -1]
        ap = float(valid.mean()) if valid.size else float("nan")
        ap50_values = precision[0, :, :, area_index, -1]
        ap50_valid = ap50_values[ap50_values > -1]
        rows.append(
            {
                "size": name,
                "min_area_ratio": lo,
                "max_area_ratio": hi,
                "ap50_95": ap,
                "ap50": float(ap50_valid.mean()) if ap50_valid.size else float("nan"),
                "gt_count": sum(lo <= ann["area"] < hi for ann in annotations),
            }
        )
    return rows


def plot_size_results(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [spec.label for spec in CORE_MODELS]
    sizes = [name for name, _, _ in SIZE_BINS]
    x = np.arange(len(sizes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, method in enumerate(methods):
        values = [next(row["ap50_95"] for row in rows if row["method"] == method and row["size"] == size) for size in sizes]
        ax.bar(x + (index - 1) * width, values, width, label=method)
    ax.set_xticks(x, sizes)
    ax.set_ylabel("AP50-95")
    ax.set_xlabel("Normalized target-area stratum")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def size_phase(args: argparse.Namespace) -> None:
    images, annotations, id_to_path = load_yolo_ground_truth(args)
    rows = []
    for spec in CORE_MODELS:
        predictions = collect_predictions(spec, args, id_to_path)
        for row in coco_ap_by_size(images, annotations, predictions):
            rows.append({"method_key": spec.key, "method": spec.label} | row)
    counts = []
    for name, lo, hi in SIZE_BINS:
        count = sum(lo <= ann["area"] < hi for ann in annotations)
        counts.append({"size": name, "count": count, "fraction": count / len(annotations), "min": lo, "max": hi})
    out = Path(args.output).resolve() / "size_stratified"
    write_csv(out / "size_ap.csv", rows)
    write_csv(out / "target_distribution.csv", counts)
    plot_size_results(rows, out / "size_ap")


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "p05": float(torch.quantile(x.flatten(), 0.05)),
        "p50": float(torch.quantile(x.flatten(), 0.50)),
        "p95": float(torch.quantile(x.flatten(), 0.95)),
    }


class MechanismCapture:
    def __init__(self, detector: YOLO):
        self.detector = detector
        self.dsa = next((m for m in detector.model.modules() if m.__class__.__name__ in {"P4P3_FDSA", "P4P3_R16_ScaleAttn"}), None)
        self.scfr = next((m for m in detector.model.modules() if m.__class__.__name__ == "HFP_SCR_Gate"), None)
        if self.dsa is None or self.scfr is None:
            raise RuntimeError("FDSA checkpoint does not contain P4P3_R16_ScaleAttn and HFP_SCR_Gate.")
        self.current: dict[str, Any] = {}
        self.keep_maps = False
        self.handles = [
            self.dsa.register_forward_pre_hook(self._capture_dsa),
            self.scfr.register_forward_pre_hook(self._capture_scfr),
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _capture_dsa(self, module: Any, inputs: tuple[Any, ...]) -> None:
        xs = inputs[0]
        p3, base = module._base(xs)
        feats = []
        for proj, x in zip(module.scale_proj, xs):
            feat = proj(x)
            if feat.shape[-2:] != p3.shape[-2:]:
                feat = F.interpolate(feat, size=p3.shape[-2:], mode="nearest")
            feats.append(feat)
        pooled = torch.cat([F.adaptive_avg_pool2d(feat, 1) for feat in feats], dim=1)
        weights = module.scale_logits(pooled).softmax(dim=1)
        fused = torch.zeros_like(base)
        for i, feat in enumerate(feats):
            fused = fused + weights[:, i : i + 1] * feat
        scale_residual = module.scale_out(fused - base)
        a_s = 0.75 + 0.5 * scale_residual.sigmoid()
        r_p = module.weight(base) * module.refine(base)
        correction = module.alpha.tanh() * a_s * r_p
        stats = {f"source_weight_{name}": float(weights[:, i].mean()) for i, name in enumerate(SOURCE_NAMES)}
        stats |= {f"a_s_{key}": value for key, value in tensor_stats(a_s).items()}
        stats["a_s_near_lower_fraction"] = float((a_s <= 0.755).float().mean())
        stats["a_s_near_upper_fraction"] = float((a_s >= 1.245).float().mean())
        stats["pfm_correction_relative_l2"] = float(correction.norm() / (base.norm() + 1e-12))
        self.current["dsa_stats"] = stats
        if self.keep_maps:
            self.current["dsa_maps"] = {
                "a_s": a_s.mean(1)[0].detach().float().cpu().numpy(),
                "pfm_correction": correction.abs().mean(1)[0].detach().float().cpu().numpy(),
            }

    def _capture_scfr(self, module: Any, inputs: tuple[Any, ...]) -> None:
        x = module.proj(inputs[0])
        c = x.shape[1]
        kernel = module.laplacian_kernel.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        detail = F.conv2d(x, kernel, padding=1, groups=c)
        magnitude = detail.abs()
        continuity = 0.5 * (module.direction_h(detail).abs() + module.direction_v(detail).abs())
        evidence = torch.cat(
            (magnitude.mean(1, keepdim=True), magnitude.amax(1, keepdim=True), continuity.mean(1, keepdim=True)), dim=1
        )
        spatial = module.spatial_gate(evidence)
        channel = module.channel_gate(magnitude)
        route = module.route(evidence)
        residual = module.expand(
            route[:, :1] * module.local_refine(detail) + route[:, 1:2] * module.context_refine(detail)
        )
        injected = module.alpha.tanh() * spatial * channel * residual
        stats = {
            "scfr_magnitude_mean": float(magnitude.mean()),
            "scfr_continuity_mean": float(continuity.mean()),
            "scfr_spatial_gate_mean": float(spatial.mean()),
            "scfr_channel_gate_mean": float(channel.mean()),
            "scfr_route_local_mean": float(route[:, :1].mean()),
            "scfr_route_dilated_mean": float(route[:, 1:2].mean()),
            "scfr_injected_relative_l2": float(injected.norm() / (x.norm() + 1e-12)),
        }
        self.current["scfr_stats"] = stats
        if self.keep_maps:
            self.current["scfr_maps"] = {
                "magnitude": magnitude.mean(1)[0].detach().float().cpu().numpy(),
                "continuity": continuity.mean(1)[0].detach().float().cpu().numpy(),
                "spatial_gate": spatial[0, 0].detach().float().cpu().numpy(),
                "channel_gate": channel.mean(1)[0].detach().float().cpu().numpy(),
                "route_local": route[0, 0].detach().float().cpu().numpy(),
                "route_dilated": route[0, 1].detach().float().cpu().numpy(),
                "residual": injected.abs().mean(1)[0].detach().float().cpu().numpy(),
            }


def gt_image_stats(annotations: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[float]] = {}
    for ann in annotations:
        grouped.setdefault(ann["image_id"], []).append(float(ann["area"]))
    out = {}
    for image_id, values in grouped.items():
        out[image_id] = {
            "gt_count": len(values),
            "gt_mean_area_ratio": float(np.mean(values)),
            "gt_tiny_fraction": float(np.mean(np.asarray(values) < 0.00025)),
        }
    return out


def plot_mechanism_summary(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    source_cols = [f"source_weight_{name}" for name in SOURCE_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].boxplot([[row[col] for row in rows] for col in source_cols], tick_labels=SOURCE_NAMES, showfliers=False)
    axes[0].set_ylabel("Four-source softmax weight")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].hist([row["a_s_mean"] for row in rows], bins=30, alpha=0.75, label="mean(A_s)")
    axes[1].hist([row["pfm_correction_relative_l2"] for row in rows], bins=30, alpha=0.65, label="PFM relative L2")
    axes[1].legend(frameon=False)
    axes[1].set_xlabel("Per-image statistic")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_map_panel(image_path: Path, maps: dict[str, np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    items = [("input", None)] + list(maps.items())
    cols = 4
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6))
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, array) in zip(axes, items):
        if array is None:
            ax.imshow(Image.open(image_path).convert("RGB"))
        else:
            ax.imshow(array, cmap="magma")
        ax.set_title(name)
        ax.axis("off")
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def mechanism_phase(args: argparse.Namespace) -> None:
    images, annotations, id_to_path = load_yolo_ground_truth(args)
    image_stats = gt_image_stats(annotations)
    detector = YOLO(seed0_weights(model_by_key("fdsa-yolo"), args).as_posix())
    capture = MechanismCapture(detector)
    rows = []
    try:
        for image_id, image_path in id_to_path.items():
            capture.current = {}
            capture.keep_maps = False
            detector.predict(
                source=image_path.as_posix(),
                imgsz=IMGSZ,
                conf=0.25,
                device=args.analysis_device,
                batch=1,
                verbose=False,
                save=False,
            )
            if "dsa_stats" not in capture.current or "scfr_stats" not in capture.current:
                raise RuntimeError(f"Mechanism hooks did not fire for {image_path}")
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": image_path.name,
                    **image_stats.get(image_id, {"gt_count": 0, "gt_mean_area_ratio": 0.0, "gt_tiny_fraction": 0.0}),
                    **capture.current["dsa_stats"],
                    **capture.current["scfr_stats"],
                }
            )
        out = Path(args.output).resolve() / "mechanism"
        write_csv(out / "per_image_mechanism_stats.csv", rows)
        aggregate = {}
        numeric_cols = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "image_id"]
        for col in numeric_cols:
            values = np.asarray([row[col] for row in rows], dtype=float)
            aggregate[col] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
            }
        write_json(out / "aggregate_mechanism_stats.json", aggregate)
        correlations = []
        targets = ("gt_count", "gt_mean_area_ratio", "gt_tiny_fraction")
        mechanism_cols = [col for col in numeric_cols if col not in targets]
        for mechanism_col in mechanism_cols:
            x = np.asarray([row[mechanism_col] for row in rows], dtype=float)
            for target in targets:
                y = np.asarray([row[target] for row in rows], dtype=float)
                corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
                correlations.append({"mechanism_stat": mechanism_col, "target_stat": target, "pearson_r": corr})
        write_csv(out / "mechanism_correlations.csv", correlations)
        plot_mechanism_summary(rows, out / "mechanism_summary")

        ordered = sorted(rows, key=lambda row: row["a_s_std"])
        indices = [int(round((len(ordered) - 1) * q)) for q in (0.1, 0.5, 0.9)]
        representatives = [ordered[index] for index in indices]
        write_csv(out / "representative_selection.csv", representatives)
        for rank, row in enumerate(representatives, 1):
            image_path = id_to_path[int(row["image_id"])]
            capture.current = {}
            capture.keep_maps = True
            detector.predict(
                source=image_path.as_posix(),
                imgsz=IMGSZ,
                conf=0.25,
                device=args.analysis_device,
                batch=1,
                verbose=False,
                save=False,
            )
            maps = capture.current.get("dsa_maps", {}) | capture.current.get("scfr_maps", {})
            save_map_panel(image_path, maps, out / "maps" / f"case{rank}_{image_path.stem}.png")
    finally:
        capture.close()


def xywh_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_image(
    gt: list[dict[str, Any]], predictions: list[dict[str, Any]], conf: float, iou_threshold: float
) -> tuple[set[int], set[int], list[dict[str, Any]]]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    kept = [(index, pred) for index, pred in enumerate(predictions) if pred["score"] >= conf]
    for pred_index, pred in sorted(kept, key=lambda item: item[1]["score"], reverse=True):
        candidates = [
            (xywh_iou(pred["bbox"], ann["bbox"]), ann["id"])
            for ann in gt
            if ann["id"] not in matched_gt and ann["category_id"] == pred["category_id"]
        ]
        if not candidates:
            continue
        best_iou, gt_id = max(candidates)
        if best_iou >= iou_threshold:
            matched_gt.add(gt_id)
            matched_pred.add(pred_index)
    false_positives = [pred for index, pred in kept if index not in matched_pred]
    return matched_gt, matched_pred, false_positives


def draw_predictions(
    image_path: Path,
    predictions: list[dict[str, Any]],
    highlight_gt: list[dict[str, Any]] | None = None,
    highlight_color: str = "#FFD400",
    conf: float = 0.25,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    colors = ("#00A6D6", "#00B86B", "#B45F06", "#D62728", "#7A3DB8", "#008C8C", "#B8860B", "#C03A7A", "#2457C5", "#A64B00")
    line_width = max(1, round(min(width, height) / 700))
    for pred in predictions:
        if pred["score"] < conf:
            continue
        x, y, w, h = pred["bbox"]
        box = (x * width, y * height, (x + w) * width, (y + h) * height)
        cls = int(pred["category_id"]) - 1
        color = colors[cls % len(colors)]
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{CLASS_NAMES[cls]} {pred['score']:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), label, fill="white", font=font)
    if highlight_gt:
        yellow_width = max(1, line_width)
        for ann in highlight_gt:
            x, y, w, h = ann["bbox"]
            draw.rectangle(
                (x * width, y * height, (x + w) * width, (y + h) * height),
                outline=highlight_color,
                width=yellow_width,
            )
    return image


def fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return canvas


def qualitative_phase(args: argparse.Namespace) -> None:
    images, annotations, id_to_path = load_yolo_ground_truth(args)
    gt_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)
    preds_by_model: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for spec in CORE_MODELS:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for pred in collect_predictions(spec, args, id_to_path):
            grouped.setdefault(pred["image_id"], []).append(pred)
        preds_by_model[spec.key] = grouped
    candidates = []
    match_cache: dict[tuple[str, int], tuple[set[int], set[int], list[dict[str, Any]]]] = {}
    for image_id, image_path in id_to_path.items():
        gt = gt_by_image.get(image_id, [])
        for spec in CORE_MODELS:
            match_cache[(spec.key, image_id)] = match_image(
                gt, preds_by_model[spec.key].get(image_id, []), args.qual_conf, args.qual_iou
            )
        base_match = match_cache[("yolov8n", image_id)][0]
        parent_match = match_cache[("scfr-pfm", image_id)][0]
        fdsa_match = match_cache[("fdsa-yolo", image_id)][0]
        fdsa_only = fdsa_match - base_match - parent_match
        comparator_only = (base_match | parent_match) - fdsa_match
        candidates.append(
            {
                "image_id": image_id,
                "file_name": image_path.name,
                "gt_count": len(gt),
                "fdsa_only_tp": len(fdsa_only),
                "comparator_only_tp": len(comparator_only),
                "fdsa_false_positive": len(match_cache[("fdsa-yolo", image_id)][2]),
                "fdsa_only_gt_ids": sorted(fdsa_only),
                "comparator_only_gt_ids": sorted(comparator_only),
            }
        )
    success = sorted(candidates, key=lambda row: (-row["fdsa_only_tp"], row["fdsa_false_positive"], -row["gt_count"]))
    selected_success = [row for row in success if row["fdsa_only_tp"] > 0][:3]
    failure = sorted(candidates, key=lambda row: (-row["comparator_only_tp"], row["fdsa_only_tp"]))
    selected_failure = next((row for row in failure if row["comparator_only_tp"] > 0), None)
    selected = selected_success + ([selected_failure] if selected_failure else [])
    for rank, row in enumerate(selected, 1):
        row["selection_role"] = "success" if rank <= len(selected_success) else "failure"
        row["selection_rank"] = rank
        row["highlight_semantics"] = (
            "yellow: GT-matched FDSA-only true positive"
            if row["selection_role"] == "success"
            else "red: GT matched by a comparator but missed by FDSA-YOLO"
        )
    out = Path(args.output).resolve() / "qualitative"
    write_csv(out / "all_candidate_scores.csv", candidates)
    write_json(out / "selection_manifest.json", selected)

    panel_width, panel_height = 560, 340
    header_height = 42
    headers = ("Input", "YOLOv8n", "SCFR+PFM", "FDSA-YOLO", "GT-backed highlight")
    composite = Image.new("RGB", (panel_width * len(headers), header_height + panel_height * len(selected)), "white")
    draw = ImageDraw.Draw(composite)
    font = ImageFont.load_default()
    for col, header in enumerate(headers):
        draw.text((col * panel_width + 8, 14), header, fill="black", font=font)
    gt_by_id = {ann["id"]: ann for ann in annotations}
    for row_index, row in enumerate(selected):
        image_id = int(row["image_id"])
        image_path = id_to_path[image_id]
        success_case = row["selection_role"] == "success"
        highlight_ids = row["fdsa_only_gt_ids"] if success_case else row["comparator_only_gt_ids"]
        highlights = [gt_by_id[int(gt_id)] for gt_id in highlight_ids]
        panels = [
            Image.open(image_path).convert("RGB"),
            draw_predictions(image_path, preds_by_model["yolov8n"].get(image_id, []), conf=args.qual_conf),
            draw_predictions(image_path, preds_by_model["scfr-pfm"].get(image_id, []), conf=args.qual_conf),
            draw_predictions(image_path, preds_by_model["fdsa-yolo"].get(image_id, []), conf=args.qual_conf),
            draw_predictions(
                image_path,
                preds_by_model["fdsa-yolo"].get(image_id, []),
                highlight_gt=highlights,
                highlight_color="#FFD400" if success_case else "#E53935",
                conf=args.qual_conf,
            ),
        ]
        for col, panel in enumerate(panels):
            composite.paste(fit_panel(panel, panel_width, panel_height), (col * panel_width, header_height + row_index * panel_height))
    composite.save(out / "fig5_gt_backed_candidates.png", quality=95)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("dry", "seed", "seed-worker", "size", "mechanism", "qualitative", "all"),
        default="dry",
    )
    parser.add_argument("--data", default=DEFAULT_DATA.as_posix())
    parser.add_argument("--vis-root", default=DEFAULT_VIS_ROOT.as_posix())
    parser.add_argument("--paper-root", default=DEFAULT_PAPER_ROOT.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--gpus", default="0,2,3,4,5,6")
    parser.add_argument("--device", default="0", help="Worker-only CUDA id.")
    parser.add_argument("--analysis-device", default="7")
    parser.add_argument("--val-devices", default="0,2,3,4,5,6,7")
    parser.add_argument("--latency-device", default="7")
    parser.add_argument("--job", default="", help="Internal seed worker job, e.g. fdsa-yolo:1.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--qual-conf", type=float, default=0.25)
    parser.add_argument("--qual-iou", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.output).resolve().mkdir(parents=True, exist_ok=True)
    if args.phase == "dry":
        dry_check(args)
    elif args.phase == "seed-worker":
        if not args.job:
            raise ValueError("--job is required for seed-worker.")
        train_worker(args)
    elif args.phase == "seed":
        seed_phase(args)
    elif args.phase == "size":
        size_phase(args)
    elif args.phase == "mechanism":
        mechanism_phase(args)
    elif args.phase == "qualitative":
        qualitative_phase(args)
    elif args.phase == "all":
        dry_check(args)
        seed_phase(args)
        size_phase(args)
        mechanism_phase(args)
        qualitative_phase(args)
    write_json(Path(args.output).resolve() / "environment_manifest.json", environment_manifest(args))


if __name__ == "__main__":
    main()
