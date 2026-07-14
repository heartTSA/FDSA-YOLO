"""Train a released FDSA-YOLO model from YAML."""
from __future__ import annotations
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--project", default="runs/fdsa/train")
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    args = p.parse_args()
    YOLO(args.model).train(data=args.data, project=args.project, name=args.name, device=args.device,
        seed=args.seed, batch=args.batch, workers=args.workers, imgsz=args.imgsz, epochs=args.epochs,
        patience=args.patience, pretrained=True, optimizer="auto", deterministic=True, amp=True,
        plots=True, save=True, save_period=-1)

if __name__ == "__main__":
    main()
