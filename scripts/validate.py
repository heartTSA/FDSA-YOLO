"""Validate a released checkpoint and export paper-facing metrics."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from ultralytics import YOLO

def built(value):
    if isinstance(value, dict): return {str(k): built(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [built(v) for v in value]
    if hasattr(value, "item"): return value.item()
    if hasattr(value, "tolist"): return value.tolist()
    return value

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True); p.add_argument("--data", required=True)
    p.add_argument("--output", required=True); p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=56); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    result = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, half=False, plots=True, project=out.parent.as_posix(), name=out.name)
    layers, params, gradients, gflops = model.info(detailed=False, verbose=False, imgsz=args.imgsz)
    payload = {"weights": args.weights, "data": args.data, "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "layers": layers, "params": params, "params_m": params / 1e6,
        "gradients": gradients, "gflops": gflops, "results": built(getattr(result, "results_dict", {}))}
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = built(result.summary(normalize=True, decimals=8)) if hasattr(result, "summary") else []
    if rows:
        with (out / "per_class_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__":
    main()
