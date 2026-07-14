"""Build release YAMLs and optionally load the nine released checkpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser(); p.add_argument("--models-dir", default="models"); p.add_argument("--weights-dir")
    p.add_argument("--device", default="cpu"); p.add_argument("--imgsz", type=int, default=640); args=p.parse_args()
    for cfg in sorted(Path(args.models_dir).glob("*.yaml")):
        model=YOLO(cfg.as_posix()); model.info(detailed=False,verbose=False,imgsz=args.imgsz)
        detector=model.model.to(args.device).eval(); detector(torch.randn(1,3,args.imgsz,args.imgsz,device=args.device))
        print(f"OK build/forward: {cfg}")
    if args.weights_dir:
        for weight in sorted(Path(args.weights_dir).glob("*_best.pt")):
            YOLO(weight.as_posix()); print(f"OK checkpoint: {weight}")

if __name__ == "__main__": main()
