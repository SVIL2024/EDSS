# EDSS

Official PyTorch implementation of **EDSS: Evidence-Guided Dense Snippet
Supervision for Weakly Supervised Video Anomaly Detection**.

## Overview

EDSS is a training-time dense snippet supervision objective for weakly
supervised video anomaly detection. It uses exact normal-video supervision and
evidence-guided pseudo-positive selection for abnormal videos, while retaining
the original VadCLIP top-$k$ multiple-instance learning (MIL) losses.

At inference time, the standard evaluation paths use the detector's ordinary
snippet scores and repeat each score over its 16-frame span. No EDSS selector or
e-BH step is run during inference.

![EDSS method overview](paper/主框图.png)

## Repository Layout

```text
EDSS/
|-- configs/                 # Public UCF-Crime and XD-Violence launchers
|-- docs/                    # Method, results, limitations, and reproducibility
|-- list/                    # Split metadata and evaluation ground truth
|-- paper/                   # Anonymous manuscript, figures, and bibliography
|-- src/                     # Model, training, evaluation, and shared utilities
|-- tests/                   # Deterministic regression tests
|-- LICENSE
|-- requirements.txt         # Python dependencies
`-- README.md
```

## Environment

Create an environment and install the dependencies:

```bash
git clone https://github.com/SVIL2024/EDSS.git
cd EDSS
conda create -n edss python=3.10 -y
conda activate edss
python -m pip install -r requirements.txt
```

If your CUDA driver, GPU, or platform differs, install a PyTorch build that
matches your machine first, then install the remaining packages from
`requirements.txt`.

Check CUDA visibility:

```bash
python - <<'PY'
import torch

print('torch:', torch.__version__)
print('cuda build:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
PY
```

## Data Preparation

This project expects pre-extracted CLIP ViT-B/16 features and the dataset
annotations. The dataset package for both supported benchmarks is available
from Quark Drive:

### Pre-extracted Features

| Dataset | Feature backbone | Download | Extraction code |
|---|---|---|---|
| UCF-Crime | ViT-B/16-CLIP | [Quark Drive](https://pan.quark.cn/s/b57edbb83bd4) | `TwtL` |
| XD-Violence | ViT-B/16-CLIP | [Quark Drive](https://pan.quark.cn/s/b57edbb83bd4) | `TwtL` |

The package may contain raw videos, annotations, and/or pre-extracted
features. Use the feature directories supplied by the package, or prepare
compatible CLIP snippet features following the upstream VadCLIP procedure.
Dataset and feature files remain subject to their original licenses and access
rules.

Expected feature-root layout:

```text
/path/to/features/
|-- UCFClipFeatures/         # class subdirectories, e.g. Abuse/, Normal/
|-- XDTrainClipFeatures/     # flat .npy feature files
`-- XDTestClipFeatures/      # flat .npy feature files
```

### Local manifest generation

The CSV manifests contain local feature paths and are intentionally ignored by
Git. Generate them after downloading or extracting the features:

```bash
python list/make_list_ucf.py \
  --feature-root /path/to/features/UCFClipFeatures \
  --split list/Anomaly_Train.txt \
  --output list/ucf_CLIP_rgb.csv

python list/make_list_ucf.py \
  --feature-root /path/to/features/UCFClipFeatures \
  --split list/Anomaly_Test.txt \
  --indices 5 \
  --output list/ucf_CLIP_rgbtest.csv

python list/make_list_xd.py \
  --feature-root /path/to/features/XDTrainClipFeatures \
  --output list/xd_CLIP_rgb.csv

python list/make_list_xd.py \
  --feature-root /path/to/features/XDTestClipFeatures \
  --output list/xd_CLIP_rgbtest.csv
```

Split files, annotations, and compact ground-truth arrays are included under
`list/`. See [`list/README.md`](list/README.md) for the file mapping.

## Pre-trained Models

The pretrained checkpoint package is available from Quark Drive:

| Package | Download | Extraction code |
|---|---|---|
| EDSS UCF-Crime and XD-Violence checkpoints | [Quark Drive](https://pan.quark.cn/s/e835d8220645) | `JEbV` |

After downloading, place the checkpoints anywhere convenient and pass the
path through `--model-path`. Do not commit downloaded checkpoints to this
repository.

## Training

Run the paper recipes after preparing the local manifests:

UCF-Crime:

```bash
bash configs/edss_ucf.sh
```

XD-Violence:

```bash
bash configs/edss_xd.sh
```

Additional command-line options are forwarded to the corresponding training
script. For example:

```bash
bash configs/edss_ucf.sh --train-list /path/to/ucf_train.csv
```

Training outputs are written to ignored local directories such as `logs/` and
`model/`. The training code does not modify Git metadata, create commits, or
upload experiment artifacts.

## Evaluation

Standard evaluation requires a local checkpoint and generated test manifest.

UCF-Crime:

```bash
python src/ucf_test.py --model-path /abs/path/to/best_ucf.pth
```

XD-Violence:

```bash
python src/xd_test.py --model-path /abs/path/to/best_xd.pth
```

The test scripts report frame-level AUC/AP and the dataset-specific temporal
localization metrics. The UCF protocol reads the classification branch, while
the XD-Violence protocol reads the vision-language alignment branch.

## Results

The following values are the single-seed, test-best results reported in the
current manuscript:

| Method | UCF-Crime AUC (%) | XD-Violence AP (%) |
|---|---:|---:|
| VadCLIP (published reference) | 88.02 | 84.51 |
| EDSS | **89.00** | **85.76** |

The VadCLIP numbers are published reference values rather than a
same-environment re-evaluation. Available same-environment controls support a
positive UCF-Crime result in the tested setting but do not show an improvement
for the strict XD-Violence comparison. Fixed-budget selectors also exceeded
adaptive e-BH rows in the available single-seed screening; EDSS therefore does
not claim a universal adaptive-budget advantage.

## Reproducibility Notes

- The paper recipes use seed `234`.
- Each feature snippet represents 16 consecutive frames.
- EDSS is applied during training; inference uses the ordinary VadCLIP score
  paths.
- Checkpoint selection follows the best test metric observed during training.
- The public source release excludes raw videos, downloaded feature arrays,
  local CSV paths, checkpoints, logs, experiment artifacts, private notes, and
  local session material.
- Run `python -m pytest tests -q` from the repository root to execute the
  deterministic regression suite.

## Acknowledgements

This project builds on:

- [VadCLIP](https://github.com/nwpu-zxr/VadCLIP): Adapting Vision-Language
  Models for Weakly Supervised Video Anomaly Detection.
- [OpenAI CLIP](https://github.com/openai/CLIP).
- The UCF-Crime and XD-Violence benchmark and annotation releases.

Please follow the original citation and license requirements when using the
adapted implementation, datasets, features, or checkpoints.

## Citation

If you use this repository or the EDSS results, please cite the manuscript and
the repository:

```bibtex
@misc{edss2026,
  author       = {SVIL2024},
  title        = {Evidence-Guided Dense Snippet Supervision for Weakly Supervised Video Anomaly Detection},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/SVIL2024/EDSS}
}
```
