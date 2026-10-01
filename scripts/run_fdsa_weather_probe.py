#!/usr/bin/env python3
"""Evaluate fixed FDSA-YOLO checkpoints on real haze and paired VisDrone degradations."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import traceback
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from run_drones_editor_benchmarks import dataset_images_and_labels, label_for_image


ALL_JOBS = (
    "vis-yolov8n", "vis-scfr-pfm", "vis-fdsa-yolo", "vis-frfdet-t", "vis-gs-yolo-n",
    "hazy-yolov8n", "hazy-scfr-pfm", "hazy-fdsa-yolo", "hazy-frfdet-t",
)
VIS_JOBS = ALL_JOBS[:5]
HAZY_JOBS = ALL_JOBS[5:]
CLASSES = ("car", "truck", "bus")
VIS_CONDITIONS = ("clean", "dark", "blur", "fog")
SIZE_RANGES = (("all", 0, 1e10), ("tiny", 0, 250), ("small", 250, 1000),
               ("medium", 1000, 5000), ("large", 5000, 1e10))
PROTOCOL = "COCOeval-bbox normalized-1000, IoU=.50:.05:.95, conf=.001, NMS=.7, maxDet=300"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def assert_images_match(directory: Path, expected: set[str]) -> None:
    actual = {path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in
              {".jpg", ".jpeg", ".png", ".bmp"}}
    if actual != expected:
        raise ValueError(f"Image mismatch in {directory}: missing={len(expected-actual)}, extra={len(actual-expected)}")


def image_set_sha256(directory: Path, names: set[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(sha256(directory / name).encode("ascii"))
    return digest.hexdigest()


def prepare_real(args) -> dict:
    archive = args.real_world_zip.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination = args.output / "datasets" / "real_hazy_test"
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        source = json.loads(bundle.read("real_world/test_real.json"))
        categories = {int(item["id"]): str(item["name"]).lower() for item in source["categories"]}
        if set(categories.values()) != set(CLASSES):
            raise ValueError(f"Unexpected real-world categories: {categories}")
        if len(source["images"]) != 200 or len(source["annotations"]) != 5543:
            raise ValueError("Unexpected real-world test size; inspect archive before evaluating")
        names = [Path(item["file_name"]).name for item in source["images"]]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate real-world test filenames")
        corrected_images = []
        images = []
        for name in names:
            member = f"real_world/test/{name}"
            target = image_dir / name
            if not target.is_file() or target.stat().st_size != bundle.getinfo(member).file_size:
                with bundle.open(member) as source_file, target.open("wb") as output_file:
                    for block in iter(lambda: source_file.read(1024 * 1024), b""):
                        output_file.write(block)
        for item in source["images"]:
            with Image.open(image_dir / Path(item["file_name"]).name) as opened:
                actual_width, actual_height = opened.size
            if (item["width"], item["height"]) != (actual_width, actual_height):
                if (item["width"], item["height"]) != (actual_height, actual_width):
                    raise ValueError(f"Unexplained image dimensions for {item['file_name']}")
                corrected_images.append(item["file_name"])
            images.append({**item, "width": actual_width, "height": actual_height})
    assert_images_match(image_dir, set(names))
    image_map = {int(item["id"]): item for item in images}
    if len(image_map) != 200:
        raise ValueError("Duplicate real-world test image IDs")
    annotations = []
    invalid_annotation_ids = []
    for item in source["annotations"]:
        image_id = int(item["image_id"])
        if image_id not in image_map:
            raise ValueError(f"Unknown image ID {image_id}")
        category = categories[int(item["category_id"])]
        x, y, w, h = map(float, item["bbox"])
        image = image_map[image_id]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(image["width"]), x + w), min(float(image["height"]), y + h)
        if x2 <= x1 or y2 <= y1:
            invalid_annotation_ids.append(int(item["id"]))
            continue
        box = [x1, y1, x2-x1, y2-y1]
        annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                            "category_id": CLASSES.index(category) + 1,
                            "bbox": box, "area": box[2] * box[3],
                            "iscrowd": int(item.get("iscrowd", 0))})
    ground_truth = {"info": {}, "licenses": [], "images": images,
                    "annotations": annotations,
                    "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]}
    gt_path = destination / "ground_truth.json"
    save_json(gt_path, ground_truth)
    return {"key": "real_hazy_test", "image_dir": str(image_dir), "gt": str(gt_path),
            "images": 200, "boxes": len(annotations), "raw_boxes": len(source["annotations"]),
            "corrected_image_dimensions": corrected_images,
            "excluded_annotation_ids": invalid_annotation_ids,
            "source_sha256": sha256(archive),
            "gt_sha256": sha256(gt_path), "image_sha256": image_set_sha256(image_dir, set(names))}


def transform_image(image: Image.Image, condition: str) -> Image.Image:
    if condition == "dark":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.45)).enhance(0.85)
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.6))
    if condition == "fog":
        airlight = Image.new("RGB", image.size, (225, 225, 225))
        blended = Image.blend(image, airlight, alpha=0.28)
        return ImageEnhance.Brightness(ImageEnhance.Contrast(blended).enhance(0.78)).enhance(1.05)
    raise ValueError(condition)


def prepare_vis(args) -> list[dict]:
    image_root, images, names = dataset_images_and_labels(args.visdrone_yaml, "val")
    if len(images) != 548:
        raise ValueError(f"Expected 548 VisDrone validation images, found {len(images)}")
    if len({image.name for image in images}) != len(images):
        raise ValueError("Duplicate VisDrone validation filenames")
    gt = {"info": {}, "licenses": [], "images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": name} for i, name in enumerate(names)]}
    for image_id, image in enumerate(images, 1):
        with Image.open(image) as opened:
            width, height = opened.size
        gt["images"].append({"id": image_id, "file_name": image.name,
                             "width": width, "height": height})
        label = label_for_image(image)
        if not label.is_file():
            raise FileNotFoundError(label)
        for line in label.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Unexpected VisDrone label in {label}: {line}")
            cls, xc, yc, bw, bh = map(float, values)
            if int(cls) != cls or not (0 <= cls < len(names)):
                raise ValueError(f"Invalid class in {label}: {line}")
            x, y, w, h = (xc - bw / 2) * width, (yc - bh / 2) * height, bw * width, bh * height
            gt["annotations"].append({"id": len(gt["annotations"]) + 1,
                                      "image_id": image_id, "category_id": int(cls) + 1,
                                      "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    gt_path = args.output / "datasets" / "visdrone_val_ground_truth.json"
    save_json(gt_path, gt)
    data = []
    expected = {image.name for image in images}
    for condition in VIS_CONDITIONS:
        if condition == "clean":
            target = image_root
        else:
            target = args.output / "datasets" / f"vis_{condition}" / "images"
            target.mkdir(parents=True, exist_ok=True)
            for image in images:
                output = target / image.name
                if output.is_file():
                    continue
                with Image.open(image) as opened:
                    result = transform_image(opened.convert("RGB"), condition)
                    if image.suffix.lower() in {".jpg", ".jpeg"}:
                        result.save(output, quality=95)
                    else:
                        result.save(output)
        assert_images_match(target, expected)
        data.append({"key": f"vis_{condition}", "image_dir": str(target), "gt": str(gt_path),
                     "images": len(images), "boxes": len(gt["annotations"]),
                     "gt_sha256": sha256(gt_path), "image_sha256": image_set_sha256(target, expected)})
    return data


def prepare(args) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = [*prepare_vis(args), prepare_real(args)]
    old = args.output / "datasets_manifest.json"
    if old.is_file():
        previous = read_json(old)
        for earlier, current in zip(previous["datasets"], datasets):
            if (earlier["key"] != current["key"] or earlier["gt_sha256"] != current["gt_sha256"]
                    or earlier.get("image_sha256") != current["image_sha256"]):
                raise RuntimeError("Dataset changed within an existing weather-probe output directory")
    manifest = {"created": datetime.now(timezone.utc).isoformat(), "datasets": datasets,
                "degradations": {"dark": "brightness 0.45, contrast 0.85",
                                 "blur": "Gaussian radius 1.6",
                                 "fog": "RGB(225,225,225) alpha 0.28, contrast 0.78, brightness 1.05"},
                "protocol": PROTOCOL}
    save_json(old, manifest)
    return manifest


def tasks(manifest: dict) -> list[tuple[str, dict]]:
    by_key = {item["key"]: item for item in manifest["datasets"]}
    return [(job, by_key[f"vis_{condition}"]) for job in VIS_JOBS for condition in VIS_CONDITIONS] + [
        (job, by_key["real_hazy_test"]) for job in HAZY_JOBS]


def backend_home(args, job: str) -> Path:
    if "frfdet" in job:
        return args.benchmark_output / "isolated_sources" / "frfdet"
    if "gs-yolo" in job:
        return args.benchmark_output / "isolated_sources" / "gs_yolo"
    return getattr(args, "fdsa_root", None) or args.base / "code"


def prediction_path(args, job: str, dataset: dict) -> Path:
    return args.output / "predictions" / f"{job}__{dataset['key']}.json"


def load_backend_yolo(home: Path):
    home = home.resolve()
    if not (home / "ultralytics" / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing isolated backend: {home}")
    # Python puts the worker script directory ahead of PYTHONPATH.
    sys.path.insert(0, str(home))
    import ultralytics

    imported = Path(ultralytics.__file__).resolve()
    if home not in imported.parents:
        raise RuntimeError(f"Backend import escaped isolation: expected {home}, loaded {imported}")
    print(f"[backend] {imported}", flush=True)
    return ultralytics.YOLO


def worker(args) -> None:
    YOLO = load_backend_yolo(backend_home(args, args.job))

    manifest = read_json(args.output / "datasets_manifest.json")
    dataset = next(item for item in manifest["datasets"] if item["key"] == args.dataset_key)
    weights = args.benchmark_output / "runs" / args.job / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    gt = read_json(Path(dataset["gt"]))
    image_ids = {item["file_name"]: item["id"] for item in gt["images"]}
    if len(image_ids) != len(gt["images"]):
        raise ValueError("Ambiguous prediction filenames")
    model = YOLO(str(weights))
    detections = []
    seen = set()
    stream = model.predict(source=dataset["image_dir"], imgsz=640, batch=1, conf=0.001,
                           iou=0.7, max_det=300, device=0, half=False,
                           save=False, verbose=False, stream=True)
    for result in stream:
        filename = Path(result.path).name
        if filename not in image_ids or filename in seen:
            raise ValueError(f"Unexpected or repeated model output: {result.path}")
        seen.add(filename)
        image_id = image_ids[filename]
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            detections.append({"image_id": image_id, "category_id": int(box.cls.item()) + 1,
                               "bbox": [x1, y1, x2-x1, y2-y1], "score": float(box.conf.item())})
    if seen != set(image_ids):
        raise RuntimeError(f"Predicted {len(seen)} of {len(image_ids)} images")
    save_json(prediction_path(args, args.job, dataset), {
        "job": args.job, "dataset": dataset["key"], "weights_sha256": sha256(weights),
        "gt_sha256": dataset["gt_sha256"], "image_sha256": dataset["image_sha256"],
        "protocol": PROTOCOL,
        "backend_version": getattr(sys.modules["ultralytics"], "__version__", "unknown"),
        "backend_import": str(Path(sys.modules["ultralytics"].__file__).resolve()),
        "image_count": len(seen), "detections": detections,
    })


def infer(args, manifest: dict) -> None:
    for job, dataset in tasks(manifest):
        weights = args.benchmark_output / "runs" / job / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        result = prediction_path(args, job, dataset)
        if result.is_file():
            cached = read_json(result)
            if (cached.get("weights_sha256") == sha256(weights)
                    and cached.get("gt_sha256") == dataset["gt_sha256"]
                    and cached.get("image_sha256") == dataset["image_sha256"]
                    and cached.get("protocol") == PROTOCOL
                    and cached.get("image_count") == dataset["images"]):
                print(f"[skip] {job} {dataset['key']}", flush=True)
                continue
        home = backend_home(args, job)
        if not (home / "ultralytics" / "__init__.py").is_file():
            raise FileNotFoundError(f"Missing isolated backend: {home}")
        log = args.output / "logs" / f"{job}__{dataset['key']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(home)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [sys.executable, str(Path(__file__).resolve()), "--phase", "worker",
                   "--base", str(args.base), "--benchmark-output", str(args.benchmark_output),
                   "--visdrone-yaml", str(args.visdrone_yaml),
                   "--real-world-zip", str(args.real_world_zip),
                   "--output", str(args.output), "--gpu", str(args.gpu),
                   "--job", job, "--dataset-key", dataset["key"]]
        print(f"[predict] {job} {dataset['key']}", flush=True)
        with log.open("w", encoding="utf-8", errors="replace") as stream:
            code = subprocess.run(command, cwd=home, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT).returncode
        if code:
            tail = log.read_text(encoding="utf-8", errors="replace")[-16000:]
            log.with_name(log.stem + "_failure_tail.log").write_text(tail, encoding="utf-8")
            raise RuntimeError(f"Prediction failed for {job} {dataset['key']}; see {log}")
        log.unlink()


def normalized_records(ground_truth: dict, detections: list[dict]):
    metadata = {image["id"]: image for image in ground_truth["images"]}

    def normalize(image_id: int, box: list[float]) -> list[float]:
        image = metadata[image_id]
        sx, sy = 1000 / image["width"], 1000 / image["height"]
        x, y, w, h = box
        return [x*sx, y*sy, w*sx, h*sy]

    gt = {"info": {}, "licenses": [],
          "images": [{**image, "width": 1000, "height": 1000} for image in ground_truth["images"]],
          "categories": ground_truth["categories"], "annotations": []}
    for row in ground_truth["annotations"]:
        box = normalize(row["image_id"], row["bbox"])
        gt["annotations"].append({**row, "bbox": box, "area": box[2]*box[3]})
    pred = [{**row, "bbox": normalize(row["image_id"], row["bbox"])} for row in detections]
    return gt, pred


def evaluate(ground_truth: dict, detections: list[dict], detailed: bool = True) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    gt, pred = normalized_records(ground_truth, detections)
    counts = {name: 0 for name, _, _ in SIZE_RANGES}
    for row in gt["annotations"]:
        if row.get("iscrowd"):
            continue
        counts["all"] += 1
        area = row["area"]
        for name, low, high in SIZE_RANGES[1:]:
            if low <= area < high:
                counts[name] += 1
                break
    if not pred:
        return {"AP50_95": 0.0, "AP50": 0.0, "per_class": [], "by_size": [], "gt_counts": counts}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(pred)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(image["id"] for image in gt["images"])
        evaluator.params.catIds = sorted(item["id"] for item in gt["categories"])
        evaluator.params.maxDets = [1, 10, 300]
        ranges = SIZE_RANGES if detailed else SIZE_RANGES[:1]
        evaluator.params.areaRng = [[low, high] for _, low, high in ranges]
        evaluator.params.areaRngLbl = [name for name, _, _ in ranges]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]

    def mean_valid(values) -> float | None:
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else None

    metrics = {"AP50_95": mean_valid(precision[:, :, :, 0, 2]),
               "AP50": mean_valid(precision[0, :, :, 0, 2]),
               "gt_counts": counts}
    if detailed:
        metrics["per_class"] = [{"class": category["name"],
                                  "AP50_95": mean_valid(precision[:, :, index, 0, 2])}
                                 for index, category in enumerate(sorted(gt["categories"], key=lambda c: c["id"]))]
        metrics["by_size"] = [{"size": name, "gt_count": counts[name],
                                "AP50_95": mean_valid(precision[:, :, :, index, 2]),
                                "AP50": mean_valid(precision[0, :, :, index, 2])}
                               for index, (name, _, _) in enumerate(SIZE_RANGES)]
    return metrics


def bootstrap_sample(gt: dict, detections: list[dict], sampled: list[int]):
    image_map = {image["id"]: image for image in gt["images"]}
    ann_by_image = defaultdict(list)
    det_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)
    for detection in detections:
        det_by_image[detection["image_id"]].append(detection)
    sampled_gt = {"info": {}, "licenses": [], "images": [], "annotations": [],
                  "categories": gt["categories"]}
    sampled_det = []
    for new_image_id, old_image_id in enumerate(sampled, 1):
        sampled_gt["images"].append({**image_map[old_image_id], "id": new_image_id})
        for ann in ann_by_image[old_image_id]:
            sampled_gt["annotations"].append({**ann, "id": len(sampled_gt["annotations"]) + 1,
                                               "image_id": new_image_id})
        for detection in det_by_image[old_image_id]:
            sampled_det.append({**detection, "image_id": new_image_id})
    return sampled_gt, sampled_det


def report(args, manifest: dict) -> None:
    benchmark = args.benchmark_output / "benchmark_summary.csv"
    if not benchmark.is_file():
        raise FileNotFoundError("Corrected nine-model benchmark summary is missing")
    with benchmark.open(newline="", encoding="utf-8-sig") as stream:
        reference = list(csv.DictReader(stream))
    if {row["key"] for row in reference} != set(ALL_JOBS):
        raise ValueError("Benchmark must contain exactly the nine planned models")
    for row in reference:
        if not 0 <= float(row["common_map50_95"]) <= 1 or not row["params"] or not row["gflops"]:
            raise ValueError(f"Uncorrected common AP or complexity for {row['key']}")
        weights = args.benchmark_output / "runs" / row["key"] / "weights" / "best.pt"
        if sha256(weights) != row["weights_sha256"]:
            raise ValueError(f"Weight changed for {row['key']}")
    rows = []
    detections_by_key = {}
    for job, dataset in tasks(manifest):
        path = prediction_path(args, job, dataset)
        record = read_json(path)
        weights = args.benchmark_output / "runs" / job / "weights" / "best.pt"
        if (record["weights_sha256"] != sha256(weights)
                or record["gt_sha256"] != dataset["gt_sha256"]
                or record["image_sha256"] != dataset["image_sha256"]
                or record["image_count"] != dataset["images"]):
            raise ValueError(f"Stale prediction record: {path}")
        gt = read_json(Path(dataset["gt"]))
        metrics = evaluate(gt, record["detections"])
        if metrics["AP50_95"] is None or not 0 <= metrics["AP50_95"] <= 1:
            raise ValueError(f"Invalid AP: {job} {dataset['key']}")
        rows.append({"job": job, "dataset": dataset["key"], "AP50_95": metrics["AP50_95"],
                     "AP50": metrics["AP50"], "images": dataset["images"], "GT": metrics["gt_counts"]["all"],
                     "predictions": len(record["detections"]), "weights_sha256": record["weights_sha256"]})
        save_json(args.output / "metrics" / f"{job}__{dataset['key']}.json", metrics)
        if dataset["key"] == "real_hazy_test":
            detections_by_key[(job, dataset["key"])] = record["detections"]
        print(f"[score] {job} {dataset['key']}: AP={metrics['AP50_95']:.4f}", flush=True)
        del record, gt, metrics
        gc.collect()
    save_csv(args.output / "weather_summary.csv", rows)

    by_key = {(row["job"], row["dataset"]): row for row in rows}
    deltas = []
    for dataset in [f"vis_{condition}" for condition in VIS_CONDITIONS] + ["real_hazy_test"]:
        fdsa = "hazy-fdsa-yolo" if dataset == "real_hazy_test" else "vis-fdsa-yolo"
        others = HAZY_JOBS if dataset == "real_hazy_test" else VIS_JOBS
        for opponent in others:
            if opponent == fdsa:
                continue
            diff = by_key[(fdsa, dataset)]["AP50_95"] - by_key[(opponent, dataset)]["AP50_95"]
            clean_key = (fdsa, "vis_clean")
            clean_opponent = (opponent, "vis_clean")
            clean_diff = (by_key[clean_key]["AP50_95"] - by_key[clean_opponent]["AP50_95"]
                          if dataset.startswith("vis_") else None)
            deltas.append({"dataset": dataset, "FDSA_vs": opponent, "AP50_95_delta": diff,
                           "interaction_vs_clean": diff-clean_diff if dataset != "vis_clean" and clean_diff is not None else None})
    save_csv(args.output / "weather_deltas.csv", deltas)

    intervals = []
    if args.bootstrap:
        dataset = next(item for item in manifest["datasets"] if item["key"] == "real_hazy_test")
        gt = read_json(Path(dataset["gt"]))
        ids = [image["id"] for image in gt["images"]]
        rng = random.Random(20260929)
        samples = [[rng.choice(ids) for _ in ids] for _ in range(args.bootstrap)]
        save_json(args.output / "bootstrap_samples.json", {"seed": 20260929, "samples": samples})
        bootstrap_rows = []
        for index, sample in enumerate(samples):
            model_aps = {}
            for job in HAZY_JOBS:
                sample_gt, sample_det = bootstrap_sample(gt, detections_by_key[(job, "real_hazy_test")], sample)
                model_aps[job] = evaluate(sample_gt, sample_det, detailed=False)["AP50_95"]
                del sample_gt, sample_det
                gc.collect()
            for opponent in HAZY_JOBS:
                if opponent != "hazy-fdsa-yolo":
                    bootstrap_rows.append({"replicate": index, "FDSA_vs": opponent,
                                           "AP50_95_delta": model_aps["hazy-fdsa-yolo"] - model_aps[opponent]})
            if (index + 1) % 25 == 0:
                print(f"[bootstrap] {index + 1}/{args.bootstrap}", flush=True)
        save_csv(args.output / "bootstrap_differences.csv", bootstrap_rows)
        for opponent in HAZY_JOBS:
            if opponent == "hazy-fdsa-yolo":
                continue
            values = sorted(row["AP50_95_delta"] for row in bootstrap_rows if row["FDSA_vs"] == opponent)
            lower = values[int(0.025 * (len(values)-1))]
            upper = values[int(0.975 * (len(values)-1))]
            point = by_key[("hazy-fdsa-yolo", "real_hazy_test")]["AP50_95"] - by_key[(opponent, "real_hazy_test")]["AP50_95"]
            intervals.append({"FDSA_vs": opponent, "point_delta": point,
                              "lower_95": lower, "upper_95": upper,
                              "replicates": args.bootstrap, "interpretation": "paired image sampling only; not seed stability"})
        save_csv(args.output / "bootstrap_intervals.csv", intervals)

    real_parent = next((item for item in intervals if item["FDSA_vs"] == "hazy-scfr-pfm"), None)
    fog_parent = next(item for item in deltas if item["dataset"] == "vis_fog"
                      and item["FDSA_vs"] == "vis-scfr-pfm")
    synthetic_fdsa = next(row for row in reference if row["key"] == "hazy-fdsa-yolo")
    synthetic_parent = next(row for row in reference if row["key"] == "hazy-scfr-pfm")
    synthetic_parent_delta = (float(synthetic_fdsa["common_map50_95"])
                              - float(synthetic_parent["common_map50_95"]))
    credible_real_gain = bool(real_parent and real_parent["lower_95"] > 0)
    save_json(args.output / "claim_assessment.json", {
        "real_haze_parent_point_positive": by_key[("hazy-fdsa-yolo", "real_hazy_test")]["AP50_95"]
        > by_key[("hazy-scfr-pfm", "real_hazy_test")]["AP50_95"],
        "real_haze_parent_bootstrap_interval_positive": credible_real_gain,
        "synthetic_hazydet_parent_delta": synthetic_parent_delta,
        "vis_fog_parent_delta": fog_parent["AP50_95_delta"],
        "vis_fog_interaction_vs_clean": fog_parent["interaction_vs_clean"],
        "weather_resilience_supported": credible_real_gain and synthetic_parent_delta > 0
        and fog_parent["AP50_95_delta"] > 0,
        "fog_specific_interaction_supported": credible_real_gain and fog_parent["interaction_vs_clean"] > 0,
        "caution": "Image bootstrap does not measure training-seed stability; synthetic dark is not real night imagery.",
    })

    provenance = {"script_sha256": sha256(Path(__file__)), "benchmark_summary_sha256": sha256(benchmark),
                  "dataset_manifest_sha256": sha256(args.output / "datasets_manifest.json"),
                  "real_world_zip_sha256": sha256(args.real_world_zip),
                  "protocol": PROTOCOL, "bootstrap_seed": 20260929, "bootstrap_replicates": args.bootstrap,
                  "weights_sha256": {row["key"]: row["weights_sha256"] for row in reference}}
    save_json(args.output / "provenance.json", provenance)
    package = args.output / "weather_probe_download.tar.gz"
    included = [args.output / name for name in (
        "datasets_manifest.json", "weather_summary.csv", "weather_deltas.csv",
        "claim_assessment.json", "provenance.json",
        "datasets/real_hazy_test/ground_truth.json", "datasets/visdrone_val_ground_truth.json")]
    included += sorted((args.output / "metrics").glob("*.json"))
    included += sorted((args.output / "predictions").glob("*.json"))
    if args.bootstrap:
        included += [args.output / name for name in (
            "bootstrap_samples.json", "bootstrap_differences.csv", "bootstrap_intervals.csv")]
    with tarfile.open(package, "w:gz") as archive:
        for path in included:
            archive.add(path, arcname=path.relative_to(args.output).as_posix())
    (args.output / "DOWNLOAD_READY.txt").write_text("Weather probe complete\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "prepare", "infer", "report", "worker"), default="all")
    parser.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/fdsa_drones"))
    parser.add_argument("--fdsa-root", type=Path, help="Patched Ultralytics project root")
    parser.add_argument("--sources", type=Path, help="Directory containing pinned FRFDet and GS-YOLO clones")
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--visdrone-yaml", type=Path)
    parser.add_argument("--real-world-zip", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-key", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.base = args.base.resolve()
    args.benchmark_output = (args.benchmark_output or args.base / "results" / "editor_benchmark").resolve()
    args.visdrone_yaml = (args.visdrone_yaml or args.base / "code" / "visdrone_autodl.yaml").resolve()
    args.real_world_zip = (args.real_world_zip or args.base / "data" / "HazyDet" / "real_world.zip").resolve()
    args.output = (args.output or args.base / "results" / "weather_probe").resolve()
    if args.bootstrap < 0 or not args.gpu.isdigit():
        parser.error("--bootstrap must be nonnegative and --gpu must be a numeric physical GPU ID")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "worker":
        worker(args)
        return
    if args.phase in {"all", "prepare"} and not args.real_world_zip.is_file():
        raise FileNotFoundError(f"Upload real_world.zip before starting: {args.real_world_zip}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "DOWNLOAD_READY.txt").unlink(missing_ok=True)
    try:
        if args.phase == "all":
            command = [sys.executable, str(Path(__file__).with_name("run_drones_editor_benchmarks.py")),
                       "--phase", "validate", "--jobs", ",".join(ALL_JOBS),
                       "--gpus", args.gpu, "--fdsa-root", str(args.fdsa_root or args.base / "code"),
                       "--sources", str(args.sources or args.base / "code"),
                       "--hazydet-root", str(args.base / "data" / "HazyDet"),
                       "--visdrone-yaml", str(args.visdrone_yaml),
                       "--output", str(args.benchmark_output)]
            subprocess.run(command, check=True, cwd=args.base / "code")
        manifest = prepare(args) if args.phase in {"all", "prepare"} else read_json(args.output / "datasets_manifest.json")
        if args.phase in {"all", "infer"}:
            infer(args, manifest)
        if args.phase in {"all", "report"}:
            report(args, manifest)
    except BaseException:
        save_json(args.output / "failure.json", {"error": traceback.format_exc()[-16000:]})
        raise


if __name__ == "__main__":
    main()
