[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21373223.svg)](https://doi.org/10.5281/zenodo.21373223)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21369813.svg)](https://doi.org/10.5281/zenodo.21369813)
# FDSA-YOLO

Implementation of the **Frequency-Decoupled Scale Arbitration Neck** for object detection in dense drone imagery.

## Contents

- `fdsa_yolo/block.py`: SCFR, PFM, and FDSA modules.
- `models/`: SCFR, SCFR+PFM, FDSA-YOLO, and dependency-aware ablation definitions.
- `patches/`: parser and checkpoint-compatibility patch for Ultralytics 8.4.51.
- `scripts/`: training, validation, latency, evidence, ablation, and smoke-test entry points.
- `configs/visdrone.yaml.example`: VisDrone dataset configuration template.

## Installation

```bash
conda env create -f environment.yml
conda activate fdsa-yolo
python scripts/install_patch.py --ultralytics-root /path/to/ultralytics-8.4.51
pip install -e /path/to/ultralytics-8.4.51
python scripts/smoke_test.py --models-dir models --device cpu
```

The public model name is `P4P3_FDSA`. `P4P3_R16_ScaleAttn` is retained only as a compatibility alias for the archived checkpoints.

## Training and Evaluation

```bash
python scripts/train.py --model models/yolov8n_fdsa.yaml --data /path/to/visdrone.yaml --name fdsa_seed0 --seed 0 --device 0
python scripts/validate.py --weights runs/fdsa/train/fdsa_seed0/weights/best.pt --data /path/to/visdrone.yaml --output runs/fdsa/val/fdsa_seed0 --device 0
python scripts/benchmark_latency.py --weights runs/fdsa/train/fdsa_seed0/weights/best.pt --output runs/fdsa/latency/fdsa_seed0.json --device 0
```

The paper models use 640 x 640 inputs and 150 training epochs. The complete protocol is recorded in the manuscript and supplementary material.

## Dependency-Aware Ablation

The following command trains PFM-only and PFM+DSA-without-SCFR on two GPUs, then runs sequential validation and FP32 batch-1 latency measurement:

```bash
python scripts/run_drones_ablation.py \
  --phase all \
  --gpus 5,6 \
  --data /path/to/visdrone.yaml \
  --output runs/drones_ablation
```

The output is complete when `DOWNLOAD_READY.txt` is present.

## Data and checkpoints

Obtain VisDrone and UAVDT from their official providers and update the YAML paths locally. Checkpoints, source data, and experiment evidence are archived under the reproducibility-package concept DOI.

## License and citation

Code is released under AGPL-3.0-only.

## Archival Records

- Software concept DOI: https://doi.org/10.5281/zenodo.21373223
- Current software record: https://doi.org/10.5281/zenodo.22896571
- Reproducibility-package concept DOI: https://doi.org/10.5281/zenodo.21369813
- Current reproducibility record: https://doi.org/10.5281/zenodo.22896620
- Source repository: https://github.com/heartTSA/FDSA-YOLO

## Version 1.2.0

Version 1.2.0 adds the dependency-aware PFM and DSA ablations and their one-command experiment runner.
