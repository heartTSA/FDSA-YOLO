# FDSA-YOLO

Minimal implementation of the **Frequency-Decoupled Scale Arbitration Neck**. The release contains only the published SCFR, PFM, and DSA path; historical architecture searches and datasets are intentionally excluded.

## Contents

- `fdsa_yolo/block.py`: clean SCFR, PFM, and FDSA module implementation.
- `models/`: SCFR, SCFR+PFM, and FDSA-YOLO model definitions.
- `patches/`: parser and checkpoint-compatibility patch for Ultralytics 8.4.51.
- `scripts/`: training, validation, FP32 latency, evidence, and smoke-test entry points.
- `configs/visdrone.yaml.example`: dataset configuration template; no dataset files are redistributed.

## Installation

```bash
conda env create -f environment.yml
conda activate fdsa-yolo
python scripts/install_patch.py --ultralytics-root /path/to/ultralytics-8.4.51
pip install -e /path/to/ultralytics-8.4.51
python scripts/smoke_test.py --models-dir models --device cpu
```

The public model name is `P4P3_FDSA`. `P4P3_R16_ScaleAttn` is retained only as a compatibility alias for the archived checkpoints.

## Reproduction

```bash
python scripts/train.py --model models/yolov8n_fdsa.yaml --data /path/to/visdrone.yaml --name fdsa_seed0 --seed 0 --device 0
python scripts/validate.py --weights runs/fdsa/train/fdsa_seed0/weights/best.pt --data /path/to/visdrone.yaml --output runs/fdsa/val/fdsa_seed0 --device 0
python scripts/benchmark_latency.py --weights runs/fdsa/train/fdsa_seed0/weights/best.pt --output runs/fdsa/latency/fdsa_seed0.json --device 0
```

The paper models were trained from random initialization for 150 epochs at 640 pixels. Complete training, validation, augmentation, seed, and hardware protocols are provided in the manuscript and supplementary material.

## Data and checkpoints

Obtain VisDrone and UAVDT from their official providers and update the YAML paths locally. The nine `best.pt` checkpoints and paper-facing evidence are archived separately at `https://doi.org/10.5281/zenodo.21369814`. The software DOI is issued by Zenodo after the first GitHub release and is listed on the repository release page.

## License and citation

Code is released under AGPL-3.0. Complete `CITATION.cff` and repository/DOI placeholders before the public v1.0.0 release.
