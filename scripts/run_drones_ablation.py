#!/usr/bin/env python3
"""Run the dependency-aware FDSA-YOLO ablations used by the Drones revision.

The public phases are ``dry``, ``train``, ``val``, and ``all``. Training uses
one process per GPU. Validation and the standardized latency benchmark run
sequentially on the first selected GPU after both training workers finish.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


IMGSZ = 640
EPOCHS = 150
PATIENCE = 20
BATCH = 8
WORKERS = 2
SEED = 0
LATENCY_WARMUP = 50
LATENCY_ITERS = 200
FOCUS_CLASSES = {"pedestrian", "people", "bicycle", "motor"}


@dataclass(frozen=True)
class Experiment:
    key: str
    run_name: str
    method_label: str
    cfg: str
    paper_role: str

    def cfg_path(self) -> Path:
        path = Path(self.cfg)
        return path if path.is_absolute() else ROOT / path

    def run_dir(self, output: Path) -> Path:
        return output / "detect" / self.run_name

    def weights(self, output: Path, name: str = "best.pt") -> Path:
        return self.run_dir(output) / "weights" / name


EXPERIMENTS = (
    Experiment(
        key="pfm-only",
        run_name="drones-pfm-only-n-seed0",
        method_label="PFM-only",
        cfg="models/yolov8n_pfm_only.yaml",
        paper_role="dependency-ablation-pfm",
    ),
    Experiment(
        key="pfm-dsa-no-scfr",
        run_name="drones-pfm-dsa-no-scfr-n-seed0",
        method_label="PFM+DSA (without SCFR)",
        cfg="models/yolov8n_pfm_dsa_no_scfr.yaml",
        paper_role="dependency-ablation-pfm-dsa",
    ),
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def parse_gpus(value: str) -> list[str]:
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("--gpus must contain two distinct GPU ids, for example 5,6")
    return gpus


def analysis_device(args: argparse.Namespace) -> str:
    """Return the GPU used for sequential validation and latency measurement."""
    return str(args.analysis_device).strip() or parse_gpus(args.gpus)[0]


def experiment_by_key(key: str) -> Experiment:
    for experiment in EXPERIMENTS:
        if experiment.key == key:
            return experiment
    raise KeyError(f"Unknown experiment key: {key}")


def safe_rmtree(path: Path, root: Path) -> None:
    target = path.resolve()
    allowed = root.resolve()
    if target == allowed or allowed not in target.parents:
        raise RuntimeError(f"Refusing to remove path outside the output root: {target}")
    if target.exists():
        shutil.rmtree(target)


def training_artifacts(experiment: Experiment, output: Path) -> dict[str, Any]:
    run_dir = experiment.run_dir(output)
    files = {
        "best": experiment.weights(output, "best.pt"),
        "last": experiment.weights(output, "last.pt"),
        "results": run_dir / "results.csv",
        "args": run_dir / "args.yaml",
        "status": run_dir / "drones_ablation_status.json",
    }
    return {
        "paths": {name: path.as_posix() for name, path in files.items()},
        "exists": {name: path.exists() and path.stat().st_size > 0 for name, path in files.items()},
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def train_complete(experiment: Experiment, output: Path) -> bool:
    artifacts = training_artifacts(experiment, output)
    status_path = experiment.run_dir(output) / "drones_ablation_status.json"
    status = load_json(status_path)
    required = ("best", "last", "results", "args")
    return status.get("status") == "trained" and all(artifacts["exists"].get(name) for name in required)


def environment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    gpu_rows = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu_rows.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / (1024**3), 3),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "created": now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpu_rows,
        "command": sys.argv,
        "protocol": {
            "data": resolve_path(args.data).as_posix(),
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "workers": args.workers,
            "seed": args.seed,
            "optimizer": "auto",
            "initialization": "YAML random initialization",
            "train_devices": parse_gpus(args.gpus),
            "validation_device": analysis_device(args),
            "validation_batch": args.val_batch,
            "latency": {
                "precision": "FP32",
                "batch": 1,
                "warmup": args.latency_warmup,
                "iterations": args.latency_iters,
            },
        },
    }


def write_manifest(args: argparse.Namespace) -> None:
    output = resolve_path(args.output)
    data = resolve_path(args.data)
    rows = []
    for gpu, experiment in zip(parse_gpus(args.gpus), EXPERIMENTS):
        cfg = experiment.cfg_path().resolve()
        rows.append(
            {
                **asdict(experiment),
                "gpu": gpu,
                "cfg_resolved": cfg.as_posix(),
                "cfg_sha256": sha256(cfg) if cfg.exists() else "missing",
                "data": data.as_posix(),
                "data_sha256": sha256(data) if data.exists() else "missing",
                "seed": args.seed,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "patience": args.patience,
            }
        )
    write_csv(output / "experiment_manifest.csv", rows)
    write_json(
        output / "experiment_manifest.json",
        {
            "created": now(),
            "script": Path(__file__).resolve().as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "block_sha256": sha256(ROOT / "fdsa_yolo/block.py"),
            "tasks_sha256": sha256(Path(ultralytics.__file__).resolve().parent / "nn/tasks.py"),
            "experiments": rows,
        },
    )
    write_json(output / "environment_manifest.json", environment_manifest(args))


def dry_build(args: argparse.Namespace) -> None:
    output = resolve_path(args.output)
    data = resolve_path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    data_cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    names = data_cfg.get("names", {})
    nc = int(data_cfg.get("nc") or len(names))
    if nc <= 0:
        raise ValueError(f"Unable to determine class count from dataset YAML: {data}")
    rows = []
    failures = []
    for experiment in EXPERIMENTS:
        cfg = experiment.cfg_path().resolve()
        model = None
        try:
            model = DetectionModel(cfg.as_posix(), nc=nc, verbose=False)
            info = model.info(detailed=False, verbose=True, imgsz=args.imgsz)
            layers, params, gradients, gflops = info if info else (None, None, None, None)
            rows.append(
                {
                    "key": experiment.key,
                    "run_name": experiment.run_name,
                    "method": experiment.method_label,
                    "cfg": cfg.as_posix(),
                    "layers": layers,
                    "params": params,
                    "params_m": round(float(params) / 1e6, 6) if params else None,
                    "gradients": gradients,
                    "gflops": gflops,
                    "build_ok": True,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "key": experiment.key,
                    "run_name": experiment.run_name,
                    "method": experiment.method_label,
                    "cfg": cfg.as_posix(),
                    "build_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failures.append(experiment.key)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_csv(output / "dry_build_summary.csv", rows)
    write_json(output / "dry_build_summary.json", rows)
    print("\nDry-build summary")
    for row in rows:
        print(
            f"  {row['method']}: ok={row['build_ok']}, "
            f"params_m={row.get('params_m', '')}, gflops={row.get('gflops', '')}"
        )
    if failures:
        raise RuntimeError("Dry build failed for: " + ", ".join(failures))


def worker_status_path(experiment: Experiment, output: Path) -> Path:
    return experiment.run_dir(output) / "drones_ablation_status.json"


def train_worker(args: argparse.Namespace) -> None:
    experiment = experiment_by_key(args.job)
    output = resolve_path(args.output)
    data = resolve_path(args.data)
    run_dir = experiment.run_dir(output)
    if args.force:
        safe_rmtree(run_dir, output / "detect")
    if train_complete(experiment, output):
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    last = experiment.weights(output, "last.pt")
    status = {
        "key": experiment.key,
        "run_name": experiment.run_name,
        "method": experiment.method_label,
        "cfg": experiment.cfg_path().resolve().as_posix(),
        "data": data.as_posix(),
        "device": args.device,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "optimizer": "auto",
        "initialization": "YAML random initialization; no checkpoint transfer",
        "started": now(),
        "status": "running",
        "resume_from": last.as_posix() if last.exists() else None,
    }
    write_json(worker_status_path(experiment, output), status)
    model = None
    try:
        if last.exists():
            model = YOLO(last.as_posix())
            model.train(resume=True, device=args.device, workers=args.workers, plots=False)
        else:
            model = YOLO(experiment.cfg_path().resolve().as_posix())
            model.train(
                data=data.as_posix(),
                imgsz=args.imgsz,
                epochs=args.epochs,
                patience=args.patience,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                seed=args.seed,
                deterministic=True,
                optimizer="auto",
                pretrained=False,
                half=False,
                amp=True,
                cache=False,
                plots=False,
                project=(output / "detect").as_posix(),
                name=experiment.run_name,
                exist_ok=True,
                resume=False,
            )
        status["status"] = "trained"
        status["finished"] = now()
        status["artifacts"] = training_artifacts(experiment, output)
        required = ("best", "last", "results", "args")
        if not all(status["artifacts"]["exists"].get(name) for name in required):
            raise RuntimeError("Training returned without all required artifacts.")
        write_json(worker_status_path(experiment, output), status)
    except BaseException as exc:
        status["status"] = "failed"
        status["finished"] = now()
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["artifacts"] = training_artifacts(experiment, output)
        write_json(worker_status_path(experiment, output), status)
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def failure_tail(temp_log: Path, target: Path, limit_bytes: int = 65536) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not temp_log.exists():
        target.write_text("No stderr output was captured.\n", encoding="utf-8")
        return
    with temp_log.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - limit_bytes))
        text = handle.read().decode("utf-8", errors="replace")
    target.write_text(text or "No stderr output was captured.\n", encoding="utf-8")


def launch_train_workers(args: argparse.Namespace) -> None:
    output = resolve_path(args.output)
    scheduler = output / "scheduler"
    scheduler.mkdir(parents=True, exist_ok=True)
    processes = []
    rows = []
    for gpu, experiment in zip(parse_gpus(args.gpus), EXPERIMENTS):
        cmd = [
            sys.executable,
            Path(__file__).resolve().as_posix(),
            "--phase",
            "train-worker",
            "--job",
            experiment.key,
            "--device",
            gpu,
            "--gpus",
            args.gpus,
            "--data",
            resolve_path(args.data).as_posix(),
            "--output",
            output.as_posix(),
            "--imgsz",
            str(args.imgsz),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
        ]
        if args.force:
            cmd.append("--force")
        temp_log = scheduler / f"{experiment.run_name}_stderr.tmp"
        handle = temp_log.open("w", encoding="utf-8", errors="replace")
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        process = subprocess.Popen(
            cmd,
            cwd=ROOT.as_posix(),
            stdout=subprocess.DEVNULL,
            stderr=handle,
            env=env,
        )
        processes.append((gpu, experiment, process, temp_log, handle))
        print(f"[train] GPU {gpu}: {experiment.method_label}")

    failures = []
    for gpu, experiment, process, temp_log, handle in processes:
        return_code = process.wait()
        handle.close()
        row = {
            "gpu": gpu,
            "key": experiment.key,
            "run_name": experiment.run_name,
            "method": experiment.method_label,
            "return_code": return_code,
            "status": "ok" if return_code == 0 else "failed",
        }
        if return_code == 0:
            temp_log.unlink(missing_ok=True)
        else:
            tail_path = scheduler / f"{experiment.run_name}_failure_tail.log"
            failure_tail(temp_log, tail_path)
            temp_log.unlink(missing_ok=True)
            row["failure_tail"] = tail_path.as_posix()
            failures.append(f"{experiment.key}@GPU{gpu}(exit={return_code})")
        rows.append(row)
    write_csv(scheduler / "parallel_train_results.csv", rows)
    write_json(scheduler / "parallel_train_results.json", rows)
    if failures:
        raise RuntimeError("Training workers failed: " + ", ".join(failures))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def benchmark_latency(weights: Path, device_id: str, args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The standard latency benchmark requires CUDA.")
    device = torch.device(f"cuda:{device_id}")
    detector = YOLO(weights.as_posix()).model.to(device).float().eval()
    image = torch.randn(1, 3, args.imgsz, args.imgsz, device=device, dtype=torch.float32)
    durations = []
    with torch.inference_mode():
        for _ in range(args.latency_warmup):
            detector(image)
        torch.cuda.synchronize(device)
        for _ in range(args.latency_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            detector(image)
            end.record()
            torch.cuda.synchronize(device)
            durations.append(float(start.elapsed_time(end)))
    median_ms = statistics.median(durations)
    result = {
        "weights": weights.as_posix(),
        "weights_sha256": sha256(weights),
        "protocol": "single-A10 FP32 batch-1 forward throughput",
        "device": str(device),
        "imgsz": args.imgsz,
        "batch": 1,
        "warmup": args.latency_warmup,
        "iterations": args.latency_iters,
        "median_ms": median_ms,
        "p90_ms": percentile(durations, 0.9),
        "fps": 1000.0 / median_ms,
    }
    del detector, image
    gc.collect()
    torch.cuda.empty_cache()
    return result


def per_class_rows(results: Any) -> list[dict[str, Any]]:
    rows = jsonable(results.summary(normalize=True, decimals=8)) if hasattr(results, "summary") else []
    normalized = []
    for row in rows:
        precision = row.get("Box-P")
        recall = row.get("Box-R")
        map50 = row.get("mAP50")
        map95 = row.get("mAP50-95")
        f1 = row.get("Box-F1")
        if f1 is None and precision is not None and recall is not None:
            total = float(precision) + float(recall)
            f1 = 2.0 * float(precision) * float(recall) / total if total else 0.0
        normalized.append(
            {
                "class": row.get("Class", ""),
                "images": row.get("Images"),
                "instances": row.get("Instances"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": map50,
                "map50_95": map95,
            }
        )
    return normalized


def validate_one(experiment: Experiment, device_id: str, args: argparse.Namespace) -> dict[str, Any]:
    output = resolve_path(args.output)
    weights = experiment.weights(output)
    if not weights.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")
    save_dir = output / "val" / f"val_{experiment.run_name}"
    summary_path = save_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return load_json(summary_path)
    if args.force:
        safe_rmtree(save_dir, output / "val")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(device_id))
        torch.cuda.empty_cache()

    model = YOLO(weights.as_posix())
    info = model.info(detailed=False, verbose=False, imgsz=args.imgsz)
    layers, params, gradients, gflops = info if info else (None, None, None, None)
    results = model.val(
        data=resolve_path(args.data).as_posix(),
        imgsz=args.imgsz,
        batch=args.val_batch,
        workers=args.workers,
        device=device_id,
        half=False,
        plots=False,
        save_json=False,
        project=(output / "val").as_posix(),
        name=f"val_{experiment.run_name}",
        exist_ok=True,
    )
    metrics = jsonable(dict(getattr(results, "results_dict", {}) or {}))
    speed = jsonable(dict(getattr(results, "speed", {}) or {}))
    classes = per_class_rows(results)
    focus = [row for row in classes if str(row.get("class", "")).lower() in FOCUS_CLASSES]
    del results, model
    gc.collect()
    torch.cuda.empty_cache()
    latency = benchmark_latency(weights, device_id, args)

    summary = {
        "key": experiment.key,
        "run_name": experiment.run_name,
        "method": experiment.method_label,
        "paper_role": experiment.paper_role,
        "seed": args.seed,
        "weights": weights.as_posix(),
        "weights_sha256": sha256(weights),
        "cfg": experiment.cfg_path().resolve().as_posix(),
        "cfg_sha256": sha256(experiment.cfg_path().resolve()),
        "data": resolve_path(args.data).as_posix(),
        "imgsz": args.imgsz,
        "train_batch": args.batch,
        "val_batch": args.val_batch,
        "device": device_id,
        "layers": layers,
        "params": params,
        "params_m": float(params) / 1e6 if params else None,
        "gradients": gradients,
        "gflops": gflops,
        "precision": metrics.get("metrics/precision(B)"),
        "recall": metrics.get("metrics/recall(B)"),
        "map50": metrics.get("metrics/mAP50(B)"),
        "map50_95": metrics.get("metrics/mAP50-95(B)"),
        "fitness": metrics.get("fitness"),
        "speed_ms": speed,
        "latency": latency,
        "validated": now(),
    }
    if focus:
        for metric in ("precision", "recall", "map50", "map50_95"):
            values = [float(row[metric]) for row in focus if row.get(metric) is not None]
            summary[f"focus_{metric}"] = statistics.mean(values) if values else None
        summary["focus_classes"] = ",".join(str(row["class"]) for row in focus)

    write_json(save_dir / "summary.json", summary)
    write_csv(save_dir / "summary.csv", [flatten_summary(summary)])
    write_json(save_dir / "per_class_summary.json", classes)
    write_csv(save_dir / "per_class_summary.csv", classes)
    write_csv(save_dir / "focus_classes_summary.csv", focus)
    write_json(save_dir / "latency_summary.json", latency)
    write_csv(save_dir / "latency_summary.csv", [latency])
    return summary


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    latency = summary.get("latency", {}) or {}
    speed = summary.get("speed_ms", {}) or {}
    return {
        key: value
        for key, value in summary.items()
        if key not in {"latency", "speed_ms"} and not isinstance(value, (dict, list))
    } | {
        "speed_preprocess_ms": speed.get("preprocess"),
        "speed_inference_ms": speed.get("inference"),
        "speed_postprocess_ms": speed.get("postprocess"),
        "latency_median_ms": latency.get("median_ms"),
        "latency_p90_ms": latency.get("p90_ms"),
        "latency_fps": latency.get("fps"),
    }


def validate_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    device_id = analysis_device(args)
    rows = []
    for experiment in EXPERIMENTS:
        print(f"[validate] GPU {device_id}: {experiment.method_label}")
        rows.append(validate_one(experiment, device_id, args))
    return rows


def aggregate(args: argparse.Namespace) -> None:
    output = resolve_path(args.output)
    rows = []
    missing = []
    for experiment in EXPERIMENTS:
        path = output / "val" / f"val_{experiment.run_name}" / "summary.json"
        summary = load_json(path)
        if not summary:
            missing.append(path.as_posix())
        else:
            rows.append(flatten_summary(summary))
    if missing:
        raise RuntimeError("Missing validation summaries:\n  " + "\n  ".join(missing))
    summary_dir = output / "summary"
    write_csv(summary_dir / "drones_ablation_summary.csv", rows)
    write_json(summary_dir / "drones_ablation_summary.json", rows)
    checks = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "DOWNLOAD_READY.txt"):
        checks.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(output / "sha256_manifest.csv", checks)
    write_json(output / "sha256_manifest.json", checks)


def write_ready_marker(args: argparse.Namespace) -> None:
    output = resolve_path(args.output)
    marker = output / "DOWNLOAD_READY.txt"
    lines = [
        "FDSA-YOLO Drones dependency-aware ablation package",
        f"completed: {now()}",
        f"output: {output.as_posix()}",
        "experiments: PFM-only; PFM+DSA without SCFR",
        "status: training, clean validation, per-class metrics, and latency benchmark completed",
        "download: copy this entire directory",
    ]
    marker.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parent(args: argparse.Namespace) -> None:
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "DOWNLOAD_READY.txt"
    marker.unlink(missing_ok=True)
    if args.phase == "val" and (output / "experiment_manifest.json").exists():
        write_json(output / "validation_environment_manifest.json", environment_manifest(args))
    else:
        write_manifest(args)
    status = {
        "phase": args.phase,
        "status": "running",
        "started": now(),
        "output": output.as_posix(),
    }
    write_json(output / "pipeline_status.json", status)
    try:
        if args.phase in {"dry", "all"}:
            dry_build(args)
        if args.phase in {"train", "all"}:
            launch_train_workers(args)
        if args.phase in {"val", "all"}:
            validate_all(args)
            aggregate(args)
            write_ready_marker(args)
        status["status"] = "complete"
        status["finished"] = now()
        write_json(output / "pipeline_status.json", status)
    except BaseException as exc:
        status["status"] = "failed"
        status["finished"] = now()
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "pipeline_status.json", status)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="all", choices=["dry", "train", "val", "all", "train-worker"])
    parser.add_argument("--gpus", default="5,6", help="Two distinct GPU ids; first GPU is used for sequential validation.")
    parser.add_argument("--analysis-device", default="", help="GPU for sequential validation and latency; defaults to the first --gpus id.")
    parser.add_argument("--device", default="", help=argparse.SUPPRESS)
    parser.add_argument("--job", default="", help=argparse.SUPPRESS)
    parser.add_argument("--data", default="datasets/vis.yaml")
    parser.add_argument("--output", default="runs/drones_ablation")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--val-batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--latency-warmup", type=int, default=LATENCY_WARMUP)
    parser.add_argument("--latency-iters", type=int, default=LATENCY_ITERS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "train-worker":
        if not args.job or not args.device:
            raise ValueError("Internal train-worker requires --job and --device.")
        train_worker(args)
        return
    parse_gpus(args.gpus)
    run_parent(args)


if __name__ == "__main__":
    main()
