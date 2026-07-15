"""Install FDSA source and apply the Ultralytics 8.4.51 parser patch."""
from __future__ import annotations
import argparse, shutil, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--ultralytics-root", required=True, type=Path); args=p.parse_args()
    release=Path(__file__).resolve().parents[1]; target=args.ultralytics_root.resolve()
    if not (target/"ultralytics/nn/tasks.py").exists(): raise FileNotFoundError("Not an Ultralytics source root")
    destination=target/"fdsa_yolo"
    if destination.exists(): shutil.rmtree(destination)
    shutil.copytree(release/"fdsa_yolo", destination)
    subprocess.run(["git","apply","--check",(release/"patches/ultralytics-8.4.51-fdsa.patch").as_posix()],cwd=target,check=True)
    subprocess.run(["git","apply",(release/"patches/ultralytics-8.4.51-fdsa.patch").as_posix()],cwd=target,check=True)
    print(f"Installed FDSA-YOLO into {target}")

if __name__ == "__main__": main()
