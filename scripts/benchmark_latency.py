"""Single-device FP32 batch-1 forward latency benchmark."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import torch
from ultralytics import YOLO

def percentile(values, fraction):
    values = sorted(values); return values[max(0, min(len(values)-1, math.ceil(len(values)*fraction)-1))]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--weights", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="0"); p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50); p.add_argument("--iterations", type=int, default=200)
    args=p.parse_args(); device=torch.device(args.device if args.device.startswith("cuda") else f"cuda:{args.device}")
    model=YOLO(args.weights).model.to(device).float().eval(); image=torch.randn(1,3,args.imgsz,args.imgsz,device=device)
    times=[]
    with torch.inference_mode():
        for _ in range(args.warmup): model(image)
        torch.cuda.synchronize(device)
        for _ in range(args.iterations):
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(); model(image); end.record(); torch.cuda.synchronize(device); times.append(float(start.elapsed_time(end)))
    median=percentile(times,.5); p90=percentile(times,.9)
    row={"weights":args.weights,"device":str(device),"precision":"FP32","batch":1,"imgsz":args.imgsz,
        "warmup":args.warmup,"iterations":args.iterations,"median_ms":median,"p90_ms":p90,"fps":1000.0/median}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(row,indent=2),encoding="utf-8")
    with out.with_suffix(".csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerow(row)

if __name__ == "__main__": main()
