#!/usr/bin/env python3
"""Run the Drones editorial benchmarks without modifying the FDSA source tree.

Copy the pinned FRFDet and GS-YOLO repositories to --sources first. HazyDet
must be downloaded from its official release and supplied via --hazydet-root.
The public phases are dry, all, resume, validate, and report. The worker phase is internal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


SCRIPT_ROOT = Path(__file__).resolve().parent
CLASSES = ("car", "truck", "bus")
SOURCE_COMMITS = {
    "FRFDet": "d424df831da98f0184a8316e73b545add2b0f7a5",
    "GS-YOLO": "b6e72bf21075a037f1962cc0412cb4faf7047667",
}
FRFDET_LOCAL_HUB = '''"""Offline compatibility for the HUB package omitted from this FRFDet fork."""

HUB_WEB_ROOT = "https://hub.ultralytics.com"
PREFIX = "HUB: "


class HUBTrainingSession:
    @classmethod
    def create_session(cls, *args, **kwargs):
        raise RuntimeError("Ultralytics HUB is unavailable in this local FRFDet reproduction")


def events(*args, **kwargs):
    return None
'''
FRFDET_RAYTUNE_SHA256 = "3c0f4597d555bc6343cc8ab0b7d41428a18e52ab52820b21a9535d25bd288340"
FRFDET_NO_RAYTUNE = '"""Ray Tune reporting is disabled for standalone benchmark training."""\n\ncallbacks = {}\n'
GS_DINO_SHA256 = "3951103baf23017b35e90802758fca39f45c0ada74900614105e587680c6a069"
GS_DINO_FP32_PATCHES = (
    ("LoG = self.LoG(x_fp32)",
     "LoG = F.conv2d(x_fp32, self.LoG.weight.float(), padding=self.kernel_size // 2, "
     "groups=self.LoG.groups)\n            LoG = LoG.to(self.norm1.weight.dtype)"),
    ("edges_o = self.gaussian_conv(x_fp32)",
     "edges_o = F.conv2d(x_fp32, self.gaussian_conv.weight.float(), padding=self.size // 2, "
     "groups=self.gaussian_conv.groups)"),
)


@dataclass(frozen=True)
class Job:
    key: str
    method: str
    dataset: str
    backend: str
    config: str


JOBS = (
    Job("vis-frfdet-t", "FRFDet-T", "visdrone", "frfdet", "FRFDet-n.yaml"),
    Job("vis-gs-yolo-n", "GS-YOLO-n", "visdrone", "gs", "yolov8n-gs.yaml"),
    Job("hazy-yolov8n", "YOLOv8n", "hazydet", "fdsa", "ultralytics/cfg/models/v8/yolov8.yaml"),
    Job("hazy-scfr-pfm", "SCFR+PFM", "hazydet", "fdsa", "ultralytics/cfg/models/my/neck/yolov8-sf-e24-prefusion-frmlite-p3.yaml"),
    Job("hazy-fdsa-yolo", "FDSA-YOLO", "hazydet", "fdsa", "ultralytics/cfg/models/my/neck/yolov8-sf-r16-scaleattn-p3.yaml"),
    Job("hazy-frfdet-t", "FRFDet-T", "hazydet", "frfdet", "FRFDet-n.yaml"),
)

REFERENCE_JOBS = (
    Job("vis-yolov8n", "YOLOv8n", "visdrone", "fdsa", "ultralytics/cfg/models/v8/yolov8.yaml"),
    Job("vis-scfr-pfm", "SCFR+PFM", "visdrone", "fdsa", "ultralytics/cfg/models/my/neck/yolov8-sf-e24-prefusion-frmlite-p3.yaml"),
    Job("vis-fdsa-yolo", "FDSA-YOLO", "visdrone", "fdsa", "ultralytics/cfg/models/my/neck/yolov8-sf-r16-scaleattn-p3.yaml"),
)


def selected_jobs(args):
    available = {job.key: job for job in (*JOBS, *REFERENCE_JOBS)}
    keys = [job.key for job in JOBS] if args.jobs == "standard" else [key.strip() for key in args.jobs.split(",")]
    if not keys or len(keys) != len(set(keys)) or any(key not in available for key in keys):
        raise ValueError(f"--jobs must list distinct known jobs: {', '.join(available)}")
    return tuple(available[key] for key in keys)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def output_path(args):
    return Path(args.output).expanduser().resolve()


def fdsa_root(args):
    return Path(args.fdsa_root).expanduser().resolve()


def job_for(key):
    return next(job for job in (*JOBS, *REFERENCE_JOBS) if job.key == key)


def parse_gpu_ids(value):
    gpus = [part.strip() for part in value.split(",")]
    if not (1 <= len(gpus) <= 3) or len(set(gpus)) != len(gpus) or not all(gpu.isdigit() for gpu in gpus):
        raise ValueError("--gpus requires one to three distinct numeric GPU IDs")
    return gpus


def source_commit(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def ignore_source(directory, names):
    return {name for name in names if name in {".git", "__pycache__", ".pytest_cache", "runs"} or name.endswith(".pyc")}


def prepare_sources(args):
    """Stage pinned upstream code under the output drive."""
    output = output_path(args)
    sources = Path(args.sources).expanduser().resolve()
    stage = output / "isolated_sources"
    expected = {name: sources / name for name in SOURCE_COMMITS}
    for name, src in expected.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing {name} source: {src}")
        actual = source_commit(src)
        if actual and actual != SOURCE_COMMITS[name]:
            raise RuntimeError(f"{name} source commit is {actual}; expected {SOURCE_COMMITS[name]}")
    frf_package = stage / "frfdet" / "ultralytics"
    gs_root = stage / "gs_yolo"
    if not frf_package.exists():
        shutil.copytree(expected["FRFDet"], frf_package, ignore=ignore_source)
    if not gs_root.exists():
        shutil.copytree(expected["GS-YOLO"], gs_root, ignore=ignore_source)
    frf_hub = frf_package / "hub" / "__init__.py"
    if not frf_hub.exists():
        frf_hub.parent.mkdir(parents=True, exist_ok=True)
        frf_hub.write_text(FRFDET_LOCAL_HUB, encoding="utf-8")
    frf_raytune = frf_package / "utils" / "callbacks" / "raytune.py"
    if sha256(frf_raytune) == FRFDET_RAYTUNE_SHA256:
        frf_raytune.write_text(FRFDET_NO_RAYTUNE, encoding="utf-8")
    elif frf_raytune.read_text(encoding="utf-8") != FRFDET_NO_RAYTUNE:
        raise RuntimeError(f"Unexpected FRFDet Ray Tune callback content: {frf_raytune}")
    gs_dino = gs_root / "ultralytics" / "nn" / "modules" / "dino.py"
    if sha256(gs_dino) == GS_DINO_SHA256:
        source = gs_dino.read_text(encoding="utf-8")
        for old, new in GS_DINO_FP32_PATCHES:
            if source.count(old) != 1:
                raise RuntimeError(f"Expected one GS-YOLO FP32 convolution site: {old}")
            source = source.replace(old, new)
        gs_dino.write_text(source, encoding="utf-8")
    elif not all(new in gs_dino.read_text(encoding="utf-8") for _, new in GS_DINO_FP32_PATCHES):
        raise RuntimeError(f"Unexpected GS-YOLO LoG/Gaussian code: {gs_dino}")
    frf_base = frf_package / "cfg/models/FRFDet/FRFDet-mul.yaml"
    gs_base = gs_root / "ultralytics/cfg/models/v8/yolov8-gs.yaml"
    if not frf_base.exists() or not gs_base.exists():
        raise FileNotFoundError("Official model YAML missing from staged sources")
    for base, new_name in ((frf_base, "FRFDet-n.yaml"), (gs_base, "yolov8n-gs.yaml")):
        config = yaml.safe_load(base.read_text(encoding="utf-8"))
        config["scale"] = "n"
        (base.parent / new_name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(output / "source_manifest.json", {
        "created": timestamp(),
        "upstream_commits": SOURCE_COMMITS,
        "source_paths": {name: str(path) for name, path in expected.items()},
        "adaptations": [
            "Stage FRFDet's repository root as an isolated ultralytics package.",
            "Use the official GS-YOLO YAML directly because upstream train.py references an undefined model.",
            "Copy model YAML and set scale=n explicitly; do not change architecture modules.",
            "Provide a local-only ultralytics.hub shim absent from the FRFDet fork.",
            "Disable FRFDet's Ray Tune reporting callback in standalone benchmark runs.",
            "Cast GS-YOLO LoG and Gaussian convolution weights to FP32 for half-precision validation.",
        ],
        "frfdet_hub_shim_sha256": sha256(frf_hub),
        "frfdet_raytune_callback_sha256": sha256(frf_raytune),
        "gs_dino_fp32_sha256": sha256(gs_dino),
        "yaml_sha256": {str(path): sha256(path) for path in (
            frf_base, frf_base.parent / "FRFDet-n.yaml", gs_base, gs_base.parent / "yolov8n-gs.yaml"
        )},
    })


def prepare_hazydet(args):
    """Convert official COCO boxes and link hazy images into a YOLO dataset."""
    root = Path(args.hazydet_root).expanduser().resolve()
    target = output_path(args) / "datasets" / "hazydet"
    if not root.exists():
        raise FileNotFoundError(f"Download official HazyDet first: {root}")
    prior = read_json(target / "conversion_manifest.json")
    yaml_path = target / "hazydet.yaml"
    if prior.get("source") == str(root) and prior.get("layout_version") == 2 and yaml_path.exists():
        expected = prior.get("splits", [])
        if len(expected) == 3 and all(
            (target / "images" / row["split"]).is_dir()
            and not (target / "images" / row["split"]).is_symlink()
            and len(list((target / "labels" / row["split"]).glob("*.txt"))) == row["images"]
            and row["boxes"] > 0
            and sha256(root / row["split"] / f"{row['split']}_coco.json") == row["annotation_sha256"]
            for row in expected
        ):
            return yaml_path
    if prior and prior.get("layout_version") != 2:
        old_runs = [output_path(args) / "runs" / job.key for job in JOBS if job.dataset == "hazydet"]
        active_runs = [str(path) for path in old_runs if path.is_dir() and any(path.iterdir())]
        if active_runs:
            raise RuntimeError("HazyDet labels were not discoverable through the old directory symlinks. "
                               "Stop the current run and archive these invalid training directories before resuming: "
                               + ", ".join(active_runs))
    stats = []
    image_hashes = {}
    for split in ("train", "val", "test"):
        annotation = root / split / f"{split}_coco.json"
        images_root = root / split / "hazy_images"
        if not annotation.exists() or not images_root.is_dir():
            raise FileNotFoundError(f"Missing official HazyDet {split} images or annotations under {root}")
        doc = json.loads(annotation.read_text(encoding="utf-8"))
        categories = {int(c["id"]): str(c["name"]).strip().lower() for c in doc["categories"]}
        if set(categories.values()) != set(CLASSES):
            raise ValueError(f"Unexpected HazyDet categories: {categories}")
        img_map = {int(im["id"]): im for im in doc["images"]}
        if len(img_map) != len(doc["images"]):
            raise ValueError(f"Duplicate image IDs in {annotation}")
        filenames = [Path(str(im["file_name"])).name for im in img_map.values()]
        stems = [Path(name).stem for name in filenames]
        if len(set(filenames)) != len(filenames) or len(set(stems)) != len(stems):
            raise ValueError(f"Image filenames or stems collide in {annotation}")
        actual_images = {path.name for path in images_root.iterdir()
                         if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
        if actual_images != set(filenames):
            raise ValueError(f"COCO image list and {images_root} differ: "
                             f"missing={len(set(filenames) - actual_images)}, extra={len(actual_images - set(filenames))}")
        image_hashes[split] = set()
        label_root = target / "labels" / split
        label_root.mkdir(parents=True, exist_ok=True)
        image_link = target / "images" / split
        if image_link.is_symlink():
            image_link.unlink()
        image_link.mkdir(parents=True, exist_ok=True)
        boxes = {image_id: [] for image_id in img_map}
        for annotation_row in doc["annotations"]:
            image_id = int(annotation_row["image_id"])
            if image_id not in img_map:
                raise ValueError(f"Annotation references missing image {image_id}")
            if annotation_row.get("iscrowd", 0):
                continue
            image = img_map[image_id]
            x, y, w, h = map(float, annotation_row["bbox"])
            width, height = int(image["width"]), int(image["height"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            category = categories[int(annotation_row["category_id"])]
            cls = CLASSES.index(category)
            boxes[image_id].append(
                f"{cls} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"
            )
        for image_id, image in img_map.items():
            filename = Path(str(image["file_name"])).name
            if not (images_root / filename).is_file():
                raise FileNotFoundError(images_root / filename)
            linked_image = image_link / filename
            if not linked_image.exists():
                try:
                    linked_image.symlink_to(images_root / filename)
                except OSError:
                    shutil.copy2(images_root / filename, linked_image)
            elif linked_image.is_symlink() and linked_image.resolve() != images_root / filename:
                raise RuntimeError(f"Dataset image link points elsewhere: {linked_image}")
            image_hashes[split].add(sha256(images_root / filename))
            (label_root / (Path(filename).stem + ".txt")).write_text(
                "\n".join(boxes[image_id]) + ("\n" if boxes[image_id] else ""), encoding="utf-8"
            )
        box_count = sum(map(len, boxes.values()))
        if box_count == 0:
            raise ValueError(f"No valid HazyDet boxes in {annotation}")
        (target / "labels" / f"{split}.cache").unlink(missing_ok=True)
        stats.append({"split": split, "images": len(img_map), "boxes": box_count, "annotation_sha256": sha256(annotation)})
    overlap = (image_hashes["train"] & image_hashes["val"]) | (image_hashes["train"] & image_hashes["test"]) | (image_hashes["val"] & image_hashes["test"])
    if overlap:
        raise ValueError(f"Identical HazyDet image bytes occur in multiple splits: {len(overlap)}")
    yaml_path.write_text(yaml.safe_dump({
        "path": str(target), "train": "images/train", "val": "images/val", "test": "images/test", "names": list(CLASSES)
    }, sort_keys=False), encoding="utf-8")
    write_json(target / "conversion_manifest.json", {"created": timestamp(), "source": str(root),
                                                       "layout_version": 2, "classes": CLASSES, "splits": stats})
    return yaml_path


def prepare_smoke_dataset(args):
    """Make a fixed 20-image HazyDet subset for ten batch-2 optimizer steps."""
    output = output_path(args)
    source = output / "datasets/hazydet"
    target = output / "datasets/smoke"
    yaml_path = target / "smoke.yaml"
    if yaml_path.exists():
        return yaml_path
    for split, source_split, count in (("train", "train", 20), ("val", "val", 4)):
        images = sorted(path for path in (source / "images" / source_split).iterdir() if path.is_file())[:count]
        if len(images) != count:
            raise RuntimeError(f"Need {count} HazyDet images for smoke {split}; found {len(images)}")
        image_root = target / "images" / split
        label_root = target / "labels" / split
        image_root.mkdir(parents=True, exist_ok=True)
        label_root.mkdir(parents=True, exist_ok=True)
        for image in images:
            link = image_root / image.name
            if not link.exists():
                try:
                    link.symlink_to(image)
                except OSError:
                    shutil.copy2(image, link)
            shutil.copy2(source / "labels" / source_split / (image.stem + ".txt"), label_root / (image.stem + ".txt"))
    yaml_path.write_text(yaml.safe_dump({
        "path": str(target), "train": "images/train", "val": "images/val", "names": list(CLASSES)
    }, sort_keys=False), encoding="utf-8")
    return yaml_path


def job_paths(job, args):
    output = output_path(args)
    if job.backend == "frfdet":
        home = output / "isolated_sources" / "frfdet"
        cfg = home / "ultralytics/cfg/models/FRFDet" / job.config
    elif job.backend == "gs":
        home = output / "isolated_sources" / "gs_yolo"
        cfg = home / "ultralytics/cfg/models/v8" / job.config
    else:
        home = fdsa_root(args)
        cfg = home / job.config
        if not cfg.exists() and job.config.endswith("prefusion-frmlite-p3.yaml"):
            cfg = SCRIPT_ROOT.parent / "models/yolov8n_scfr_pfm.yaml"
        elif not cfg.exists() and job.config.endswith("r16-scaleattn-p3.yaml"):
            cfg = SCRIPT_ROOT.parent / "models/yolov8n_fdsa.yaml"
    data = Path(args.visdrone_yaml).expanduser().resolve() if job.dataset == "visdrone" else output / "datasets/hazydet/hazydet.yaml"
    run = output / "runs" / job.key
    return home, cfg, data, run


def model_config(job, cfg):
    """Use Ultralytics' virtual n-suffix to select the recorded model scale."""
    if job.backend == "fdsa":
        if not cfg.stem.startswith("yolov8"):
            raise ValueError(f"Unexpected FDSA configuration name: {cfg}")
        return cfg.with_name(cfg.stem.replace("yolov8", "yolov8n", 1) + cfg.suffix)
    return cfg


def job_status(job, args):
    return read_json(output_path(args) / "status" / (job.key + ".json"))


def current_job_status(job, args, status):
    return job.dataset != "hazydet" or status.get("dataset_layout_version") == 2


def dataset_images_and_labels(data_yaml, split):
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or Path(data_yaml).parent)
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    image_spec = cfg.get(split)
    if not isinstance(image_spec, str):
        raise ValueError(f"Common evaluation expects one {split} image directory in {data_yaml}")
    image_root = root / image_spec
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise RuntimeError(f"No {split} images in {image_root}")
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda x: int(x))]
    if not isinstance(names, list) or not names:
        raise ValueError(f"No class names in {data_yaml}")
    return image_root, images, [str(name) for name in names]


def label_for_image(image):
    parts = list(image.parts)
    indexes = [index for index, part in enumerate(parts) if part == "images"]
    if not indexes:
        raise ValueError(f"Image path lacks an images directory: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def common_coco_evaluation(model, data_yaml, split, device):
    """Evaluate all forks with the same label conversion and pycocotools code."""
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    image_root, images, names = dataset_images_and_labels(data_yaml, split)
    ground_truth = {"info": {}, "licenses": [], "images": [], "annotations": [],
                    "categories": [{"id": index + 1, "name": name} for index, name in enumerate(names)]}
    image_ids = {}
    annotation_id = 1
    for image_id, image in enumerate(images, 1):
        with Image.open(image) as img:
            width, height = img.size
        image_ids[str(image.resolve())] = image_id
        ground_truth["images"].append({"id": image_id, "file_name": image.name, "width": width, "height": height})
        label = label_for_image(image)
        if not label.exists():
            raise FileNotFoundError(label)
        for line in label.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Malformed YOLO box in {label}: {line}")
            cls, xc, yc, bw, bh = map(float, values)
            cls_int = int(cls)
            if cls != cls_int or not (0 <= cls_int < len(names)):
                raise ValueError(f"Invalid class in {label}: {line}")
            x, y, w, h = (xc - bw / 2) * width, (yc - bh / 2) * height, bw * width, bh * height
            ground_truth["annotations"].append({"id": annotation_id, "image_id": image_id,
                "category_id": cls_int + 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            annotation_id += 1
    predictions = []
    stream = model.predict(source=str(image_root), imgsz=640, batch=1,
                           conf=0.001, iou=0.7, max_det=300, device=device,
                           half=False, save=False, verbose=False, stream=True)
    unique_names = {image.name: image_ids[str(image.resolve())] for image in images}
    if len(unique_names) != len(images):
        raise ValueError(f"Duplicate image filenames in {image_root}")
    seen = set()
    for result in stream:
        image_id = image_ids.get(str(Path(result.path).resolve()))
        if image_id is None:
            image_id = unique_names.get(Path(result.path).name)
        if image_id is None:
            raise KeyError(f"Prediction path absent from dataset: {result.path}")
        seen.add(image_id)
        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            predictions.append({"image_id": image_id, "category_id": int(box.cls.item()) + 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(box.conf.item())})
    if len(seen) != len(images):
        raise RuntimeError(f"Predicted {len(seen)} of {len(images)} images from {image_root}")
    if not predictions:
        return {"map50_95": 0.0, "map50": 0.0, "images": len(images),
                "instances": len(ground_truth["annotations"]), "predictions": 0,
                "per_class": [{"class": name, "ap50_95": 0.0} for name in names],
                "protocol": "common pycocotools COCOeval; conf=0.001, NMS IoU=0.7, max_det=300"}
    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = list(range(1, len(images) + 1))
    evaluator.params.catIds = list(range(1, len(names) + 1))
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    class_rows = []
    for index, name in enumerate(names):
        category_precision = precision[:, :, index, 0, 2]
        valid = category_precision[category_precision > -1]
        class_rows.append({"class": name, "ap50_95": float(np.mean(valid)) if len(valid) else None})
    valid_precision = precision[:, :, :, 0, 2]
    valid_precision = valid_precision[valid_precision > -1]
    ap50_precision = precision[0, :, :, 0, 2]
    ap50_precision = ap50_precision[ap50_precision > -1]
    return {"map50_95": float(np.mean(valid_precision)) if len(valid_precision) else 0.0,
            "map50": float(np.mean(ap50_precision)) if len(ap50_precision) else 0.0,
            "images": len(images), "instances": len(ground_truth["annotations"]),
            "predictions": len(predictions), "per_class": class_rows,
            "protocol": "common pycocotools COCOeval; conf=0.001, NMS IoU=0.7, max_det=300"}


def validation_model_complexity(model):
    """Recover complexity when a fork's info wrapper returns no usable FLOPs."""
    params = sum(parameter.numel() for parameter in model.model.parameters())
    info = model.info(detailed=False, verbose=True)
    gflops = info[3] if info is not None else None
    method = "backend model.info; imgsz=640; FLOPs=2*MACs"
    if gflops is None or not math.isfinite(float(gflops)) or gflops <= 0:
        from copy import deepcopy
        import torch
        import thop

        try:
            probe = deepcopy(model.model).cpu().float().eval()
            with torch.inference_mode():
                macs, _ = thop.profile(probe, inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)
            gflops = 2 * float(macs) / 1e9
            method = "THOP CPU FP32 full 1x3x640x640 input; FLOPs=2*MACs"
        except Exception as exc:
            raise RuntimeError(f"640-input FLOPs profiling failed: {type(exc).__name__}: {exc}") from exc
    if params <= 0 or not math.isfinite(float(gflops)) or gflops <= 0:
        raise RuntimeError(f"Invalid model complexity: params={params}, gflops={gflops}")
    return params, float(gflops), method


def run_worker(job, args):
    home, cfg, data, run = job_paths(job, args)
    sys.path.insert(0, str(home))
    from ultralytics import YOLO
    import torch

    imported = Path(sys.modules["ultralytics"].__file__).resolve()
    if home not in imported.parents:
        raise RuntimeError(f"Backend import escaped isolation: {imported}")
    status_file = output_path(args) / "status" / (job.key + ("_smoke" if args.task == "smoke" else "") + ".json")
    previous = read_json(status_file)
    if (args.task == "train" and current_job_status(job, args, previous)
            and previous.get("state") in {"trained", "validated"} and (run / "weights" / "best.pt").exists()):
        return
    status = {"job": asdict(job), "backend_import": str(imported), "created": timestamp(), "state": "running", "cfg_sha256": sha256(cfg), "data_sha256": sha256(data)}
    if job.backend == "gs":
        status["gs_dino_fp32_sha256"] = sha256(home / "ultralytics/nn/modules/dino.py")
    if job.dataset == "hazydet":
        status["dataset_layout_version"] = read_json(output_path(args) / "datasets/hazydet/conversion_manifest.json").get("layout_version")
    write_json(status_file, status)
    try:
        if args.task == "smoke":
            model = YOLO(str(model_config(job, cfg)))
            model.model.eval()
            with torch.inference_mode():
                model.model(torch.zeros(1, 3, 64, 64))
            if args.smoke_steps:
                model.train(data=str(output_path(args) / "datasets/smoke/smoke.yaml"),
                            imgsz=640, epochs=1, patience=0, batch=2, workers=0,
                            device=args.device, seed=0, deterministic=True,
                            optimizer="SGD", pretrained=False, amp=False, cache=False,
                            val=False, save=False, plots=False, close_mosaic=0,
                            project=str(output_path(args) / "smoke_runs"),
                            name=job.key, exist_ok=True)
                check = model.val(data=str(output_path(args) / "datasets/smoke/smoke.yaml"),
                                  split="val", imgsz=640, batch=1, device=args.device,
                                  workers=0, half=job.backend == "gs", plots=False, save_json=False,
                                  project=str(output_path(args) / "smoke_runs"),
                                  name=job.key + "_val", exist_ok=True)
                if not (0.0 <= float(check.box.map) <= 1.0):
                    raise RuntimeError("Smoke validation produced an invalid AP")
            status["state"] = "smoke_ok"
        elif args.task == "train":
            # The upstream AMP reference check downloads a pretrained YOLO weight.
            # Run the local probe only after select_device() has selected the physical GPU.
            if torch.cuda.is_initialized():
                raise RuntimeError("CUDA was initialized before Ultralytics selected the requested GPU")
            from ultralytics.engine import trainer as trainer_module

            def check_amp_offline(model):
                device = next(model.parameters()).device
                if device.type != "cuda":
                    return False
                if (os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.device)
                        or torch.cuda.device_count() != 1 or device.index != 0):
                    raise RuntimeError(f"GPU mapping mismatch: requested physical GPU {args.device}, "
                                       f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                                       f"visible_count={torch.cuda.device_count()}, model_device={device}")
                with torch.autocast(device_type="cuda"):
                    probe = torch.ones((16, 16), device=device) @ torch.ones((16, 16), device=device)
                if not torch.isfinite(probe).all():
                    raise RuntimeError("Local CUDA autocast probe produced non-finite values")
                status["amp_preflight"] = "local CUDA autocast probe after device selection; no weight download"
                print(f"AMP: local CUDA autocast probe passed on {device}; training with amp=True", flush=True)
                return True

            trainer_module.check_amp = check_amp_offline
            weights = run / "weights" / "best.pt"
            last = run / "weights" / "last.pt"
            if last.exists():
                model = YOLO(str(last))
                model.train(resume=True, device=args.device, workers=args.workers, plots=False)
            else:
                model = YOLO(str(model_config(job, cfg)))
                model.train(data=str(data), imgsz=640, epochs=150, patience=20,
                            batch=args.batch, workers=args.workers, device=args.device,
                            seed=0, deterministic=True, optimizer="auto", pretrained=False,
                            amp=True, cache=False, plots=False, project=str(run.parent),
                            name=run.name, exist_ok=True, resume=False)
            for name in ("best.pt", "last.pt"):
                if not (run / "weights" / name).exists():
                    raise RuntimeError(f"Training returned without {name}")
            status["state"] = "trained"
            status["best_sha256"] = sha256(weights)
            status["args_sha256"] = sha256(run / "args.yaml")
            status["actual_optimizer"] = type(model.trainer.optimizer).__name__ if model.trainer else None
        elif args.task == "validate":
            weights = run / "weights" / "best.pt"
            if not weights.exists():
                raise FileNotFoundError(weights)
            model = YOLO(str(weights))
            metric = model.val(data=str(data), split="val" if job.dataset == "visdrone" else "test",
                               imgsz=640, batch=args.val_batch, device=args.device,
                               workers=args.workers, half=False, plots=False, save_json=False,
                               project=str(output_path(args) / "validation"), name=job.key, exist_ok=True)
            params, gflops, complexity_method = validation_model_complexity(model)
            summary = {
                "job": asdict(job), "weights_sha256": sha256(weights),
                "map50": float(metric.box.map50), "map50_95": float(metric.box.map),
                "precision": float(metric.box.mp), "recall": float(metric.box.mr),
                "params": params, "gflops": gflops, "complexity_method": complexity_method,
                "backend_version": getattr(sys.modules["ultralytics"], "__version__", "unknown"),
                "classes": metric.summary(normalize=True, decimals=8) if hasattr(metric, "summary") else [],
            }
            summary["common_eval"] = common_coco_evaluation(model, data, "val" if job.dataset == "visdrone" else "test", args.device)
            write_csv(output_path(args) / "results" / (job.key + "_common_classes.csv"), summary["common_eval"]["per_class"])
            del metric, model
            torch.cuda.empty_cache()
            if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.device) or torch.cuda.device_count() != 1:
                raise RuntimeError(f"Expected only physical GPU {args.device} to be visible for latency measurement")
            model = YOLO(str(weights)).model.to("cuda:0").float().eval()
            tensor = torch.randn(1, 3, 640, 640, device="cuda:0")
            with torch.inference_mode():
                for _ in range(50):
                    model(tensor)
                torch.cuda.synchronize(0)
                times = []
                for _ in range(200):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(tensor)
                    end.record()
                    torch.cuda.synchronize(0)
                    times.append(start.elapsed_time(end))
            times.sort()
            summary["latency_median_ms"] = times[len(times) // 2]
            summary["fps"] = 1000.0 / summary["latency_median_ms"]
            write_json(output_path(args) / "results" / (job.key + ".json"), summary)
            write_csv(output_path(args) / "results" / (job.key + "_classes.csv"), summary["classes"])
            status["state"] = "validated"
            status["best_sha256"] = sha256(weights)
        else:
            raise ValueError(args.task)
    except BaseException as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()[-12000:]
        raise
    finally:
        status["updated"] = timestamp()
        write_json(status_file, status)


def run_subprocess(job, task, device, args):
    output = output_path(args)
    home, _, _, _ = job_paths(job, args)
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", "worker", "--task", task,
               "--job", job.key, "--device", str(device), "--gpus", args.gpus,
               "--fdsa-root", str(fdsa_root(args)),
               "--sources", str(Path(args.sources).resolve()), "--hazydet-root", str(Path(args.hazydet_root).resolve()),
               "--visdrone-yaml", str(Path(args.visdrone_yaml).resolve()), "--output", str(output),
               "--batch", str(args.batch), "--val-batch", str(args.val_batch), "--workers", str(args.workers)]
    command.extend(["--smoke-steps", str(args.smoke_steps)])
    log = output / "logs" / f"{job.key}_{task}.tmp"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = str(home)
    env["TMPDIR"] = str(output / "tmp")
    env["TORCH_HOME"] = str(output / "torch_cache")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    (output / "tmp").mkdir(exist_ok=True)
    stream = log.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(command, cwd=home, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return process, stream, log


def finish_subprocess(job, task, process, stream, log, args):
    code = process.wait()
    stream.close()
    if code:
        tail = log.read_text(encoding="utf-8", errors="replace")[-65536:]
        (log.parent / f"{job.key}_{task}_failure_tail.log").write_text(tail, encoding="utf-8")
    log.unlink(missing_ok=True)
    return code


def smoke(args):
    failures = []
    gpu = parse_gpu_ids(args.gpus)[0]
    for job in selected_jobs(args):
        _, cfg, data, _ = job_paths(job, args)
        done = read_json(output_path(args) / "status" / (job.key + "_smoke.json"))
        gs_source_ok = job.backend != "gs" or done.get("gs_dino_fp32_sha256") == sha256(
            output_path(args) / "isolated_sources/gs_yolo/ultralytics/nn/modules/dino.py")
        if (done.get("state") == "smoke_ok" and done.get("cfg_sha256") == sha256(cfg)
                and done.get("data_sha256") == sha256(data) and gs_source_ok):
            continue
        print(f"[smoke] {job.key}", flush=True)
        process, stream, log = run_subprocess(job, "smoke", gpu, args)
        if finish_subprocess(job, "smoke", process, stream, log, args):
            failures.append(job.key)
    if failures:
        raise RuntimeError("Smoke tests failed: " + ", ".join(failures))


def train_all(args):
    pending = [j for j in selected_jobs(args) if not (current_job_status(j, args, job_status(j, args))
                                      and job_status(j, args).get("state") in {"trained", "validated"}
                                      and (job_paths(j, args)[3] / "weights" / "best.pt").exists())]
    gpus = parse_gpu_ids(args.gpus)
    active = {}
    failures = []
    while pending or active:
        for gpu in gpus:
            if gpu not in active and pending:
                job = pending.pop(0)
                active[gpu] = (job, *run_subprocess(job, "train", gpu, args))
                print(f"[train] GPU {gpu}: {job.key}", flush=True)
        for gpu, (job, process, stream, log) in list(active.items()):
            if process.poll() is not None:
                if finish_subprocess(job, "train", process, stream, log, args):
                    failures.append(job.key)
                del active[gpu]
        if active:
            time.sleep(2)
    if failures:
        raise RuntimeError("Training failed: " + ", ".join(failures))


def validate_all(args):
    gpu = parse_gpu_ids(args.gpus)[0]
    failures = []
    for job in selected_jobs(args):
        result = read_json(output_path(args) / "results" / (job.key + ".json"))
        weights = job_paths(job, args)[3] / "weights" / "best.pt"
        common_ap = result.get("common_eval", {}).get("map50_95") if result else None
        common_ap50 = result.get("common_eval", {}).get("map50") if result else None
        if (result and weights.exists() and result.get("weights_sha256") == sha256(weights)
                and common_ap is not None and 0 <= common_ap <= 1
                and common_ap50 is not None and 0 <= common_ap50 <= 1
                and result.get("params") is not None and result.get("params") > 0
                and result.get("gflops") is not None and result.get("gflops") > 0):
            continue
        print(f"[validate] GPU {gpu}: {job.key}", flush=True)
        process, stream, log = run_subprocess(job, "validate", gpu, args)
        if finish_subprocess(job, "validate", process, stream, log, args):
            failures.append(job.key)
    if failures:
        raise RuntimeError("Validation failed: " + ", ".join(failures))


def report(args):
    output = output_path(args)
    rows = []
    for job in selected_jobs(args):
        path = output / "results" / (job.key + ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing result: {path}")
        item = read_json(path)
        common_ap = item.get("common_eval", {}).get("map50_95")
        common_ap50 = item.get("common_eval", {}).get("map50")
        if (common_ap is None or not 0 <= common_ap <= 1
                or common_ap50 is None or not 0 <= common_ap50 <= 1
                or item.get("params") is None or item.get("params") <= 0
                or item.get("gflops") is None or item.get("gflops") <= 0):
            raise ValueError(f"Incomplete or invalid validation result: {path}; "
                             f"common_ap50_95={common_ap}, common_ap50={common_ap50}, "
                             f"params={item.get('params')}, gflops={item.get('gflops')}")
        rows.append({**asdict(job), **{k: item.get(k) for k in (
            "map50_95", "map50", "precision", "recall", "params", "gflops", "fps", "latency_median_ms", "weights_sha256", "backend_version"
        )}, "common_map50_95": item.get("common_eval", {}).get("map50_95"),
            "common_map50": item.get("common_eval", {}).get("map50")})
    write_csv(output / "benchmark_summary.csv", rows)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    try:
        gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                                  capture_output=True, text=True)
        gpu_lines = gpu_info.stdout.splitlines() if gpu_info.returncode == 0 else []
    except FileNotFoundError:
        gpu_lines = []
    write_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "pip_freeze": freeze.stdout.splitlines() if freeze.returncode == 0 else [],
        "gpu_info": gpu_lines,
    })
    files = [output / name for name in ("benchmark_summary.csv", "run_manifest.json",
             "source_manifest.json", "environment.json",
             "datasets/hazydet/conversion_manifest.json", "datasets/hazydet/hazydet.yaml")]
    files += list((output / "status").glob("*.json"))
    files += list((output / "results").glob("*"))
    for job in selected_jobs(args):
        run = job_paths(job, args)[3]
        files += [run / "args.yaml", run / "results.csv"]
        files.append(run / "weights" / "best.pt")
        last = run / "weights" / "last.pt"
        if last.is_file():
            files.append(last)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Download package input missing: {path}")
    manifest = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                 "sha256": sha256(path)} for path in files]
    write_csv(output / "sha256_manifest.csv", manifest)
    download = output / "editor_benchmarks_download.tar.gz"
    with tarfile.open(download, "w:gz") as archive:
        for path in [*files, output / "sha256_manifest.csv"]:
            archive.add(path, arcname=path.relative_to(output).as_posix())
    (output / "DOWNLOAD_READY.txt").write_text("Drones editorial benchmark complete\n" + timestamp() + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dry", "all", "resume", "validate", "report", "worker"), default="all")
    parser.add_argument("--jobs", default="standard", help="Comma-separated job keys; default is the original six jobs")
    parser.add_argument("--task", choices=("smoke", "train", "validate"), help=argparse.SUPPRESS)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    parser.add_argument("--device", help=argparse.SUPPRESS)
    parser.add_argument("--gpus", default="5,6")
    parser.add_argument("--fdsa-root", default=str(SCRIPT_ROOT), help="Existing FDSA Ultralytics project root")
    parser.add_argument("--sources", required=True, help="Directory containing pinned FRFDet and GS-YOLO clones")
    parser.add_argument("--hazydet-root", required=True, help="Official HazyDet dataset root")
    parser.add_argument("--visdrone-yaml", required=True, help="Existing VisDrone data YAML on server")
    parser.add_argument("--output", default="/mnt/Data_16T/home/server/mnt/hwt/drones_editor_revision")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--val-batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke-steps", type=int, default=10, choices=(0, 10))
    return parser.parse_args()


def main():
    args = parse_args()
    parse_gpu_ids(args.gpus)
    jobs = selected_jobs(args)
    if args.phase == "worker":
        run_worker(job_for(args.job), args)
        return
    output = output_path(args)
    output.mkdir(parents=True, exist_ok=True)
    code_files = {
        "block.py": fdsa_root(args) / "ultralytics/nn/modules/block.py",
        "tasks.py": fdsa_root(args) / "ultralytics/nn/tasks.py",
        "VisDrone YAML": Path(args.visdrone_yaml).expanduser().resolve(),
    }
    code_sha = {name: sha256(path) for name, path in code_files.items()}
    protocol = {"seed": 0, "epochs": 150, "patience": 20, "imgsz": 640,
                "batch": args.batch, "optimizer": "auto", "pretrained": False,
                "gpus": args.gpus, "val_batch": args.val_batch, "workers": args.workers,
                "fdsa_root": str(fdsa_root(args)),
                "sources": str(Path(args.sources).expanduser().resolve()),
                "hazydet_root": str(Path(args.hazydet_root).expanduser().resolve()),
                "visdrone_yaml": str(Path(args.visdrone_yaml).expanduser().resolve()),
                "code_sha256": code_sha}
    old_manifest = read_json(output / "run_manifest.json")
    if old_manifest and old_manifest.get("protocol") != protocol:
        raise RuntimeError("Existing output uses a different protocol or data path; choose a new --output")
    (output / "DOWNLOAD_READY.txt").unlink(missing_ok=True)
    write_json(output / "run_manifest.json", {
        "created": old_manifest.get("created", timestamp()), "last_run": timestamp(),
        "command": sys.argv, "python": sys.version,
        "platform": platform.platform(), "script_sha256": sha256(__file__),
        "protocol": protocol,
        "jobs": [asdict(job) for job in jobs],
    })
    try:
        if args.phase != "report":
            prepare_sources(args)
            prepare_hazydet(args)
            if args.phase in {"dry", "all", "resume"}:
                prepare_smoke_dataset(args)
        if args.phase == "dry":
            smoke(args)
        elif args.phase in {"all", "resume"}:
            smoke(args)
            train_all(args)
            validate_all(args)
            report(args)
        elif args.phase == "validate":
            validate_all(args)
            report(args)
        else:
            report(args)
    except BaseException as exc:
        write_json(output / "pipeline_failure.json", {"failed": timestamp(), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-12000:]})
        raise


if __name__ == "__main__":
    main()
