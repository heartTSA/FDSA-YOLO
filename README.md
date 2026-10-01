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
- Reproducibility-package concept DOI: https://doi.org/10.5281/zenodo.21369813
- Source repository: https://github.com/heartTSA/FDSA-YOLO

## Recent-Method and Weather Evaluation

`scripts/run_drones_editor_benchmarks.py` runs VisDrone and HazyDet training, validation, common COCO evaluation, and latency measurement. `scripts/run_fdsa_weather_probe.py` evaluates existing checkpoints on clean, synthetic-degraded, and real-haze images and reports paired image-bootstrap intervals.

Clone the comparison sources at the recorded commits:

```bash
git clone https://github.com/HZAI-ZJNU/FRFDet sources/FRFDet
git -C sources/FRFDet checkout d424df831da98f0184a8316e73b545add2b0f7a5
git clone https://github.com/bearono-s/GS-YOLO sources/GS-YOLO
git -C sources/GS-YOLO checkout b6e72bf21075a037f1962cc0412cb4faf7047667
python scripts/run_drones_editor_benchmarks.py --phase all --gpus 0 --fdsa-root /path/to/patched-ultralytics --sources sources --hazydet-root /path/to/HazyDet --visdrone-yaml /path/to/visdrone.yaml --output /path/to/results/editor_benchmark
```

Obtain HazyDet from [the official project](https://github.com/GrokCV/HazyDet). Common evaluation uses confidence 0.001, NMS IoU 0.7, and maxDet 300. Source-data files include native metrics, training settings, framework versions, and scoring audits. New throughput measurements use RTX 4080 SUPER FP32 batch 1; original A10 measurements remain in the earlier records.

For weather evaluation, place `real_world.zip` at `/path/to/base/data/HazyDet/real_world.zip`, retain the benchmark output, and provide the VisDrone YAML:

```bash
python scripts/run_fdsa_weather_probe.py --phase all --base /path/to/base --gpu 0 --fdsa-root /path/to/patched-ultralytics --sources sources --benchmark-output /path/to/results/editor_benchmark --visdrone-yaml /path/to/visdrone.yaml
```

## Version 1.3.1

Adds recent-method benchmarking, weather evaluation, common COCO scoring, and paired-bootstrap reporting.

