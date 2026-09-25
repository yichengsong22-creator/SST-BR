# SST-BR

**Spatial–Spectral–Temporal Baseline–Residual Learning for mmWave Radar Heart-Rate Estimation**

## Overview

This repository contains the complete implementation used to reproduce the
SST-BR experiment on the RHB dataset. It includes raw radar and PPG loading,
PPG reference generation, spatial–spectral feature extraction, long-time
temporal fusion, baseline–residual regression, four-fold subject-disjoint
evaluation, and pooled 10-s heart-rate estimation.

The full reproduction entry point is:

```bash
python scripts/run_full_pipeline.py --config configs/rhb_full.yaml
```

## Full Pipeline at a Glance

```text
Raw RHB radar + PPG
        ↓
Dataset manifest and PPG reference generation
        ↓
5-s local radar windows
        ↓
Spatial–spectral feature extraction
        ↓
Long-time temporal fusion
        ↓
Baseline–residual regression
        ↓
Local HR predictions
        ↓
Two adjacent local predictions → one 10-s estimate
        ↓
Four-fold pooled evaluation
```

## Installation

The release uses Python 3.11 and the pinned dependencies in
`requirements.txt`. From the repository root:

```bash
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/verify_installation.py
```

The verification script checks the installed package, full-pipeline imports,
the ordered 331-D feature contract, and the four public fold definitions. It
does not require the RHB dataset.

## RHB Dataset

The experiments use the RHB dataset introduced and released with
[Radar-APLANC](https://github.com/RadarHRSensing/Radar-APLANC). The dataset was
not collected by the SST-BR authors and is not stored in this Git repository.

**Download the RHB dataset used by this release:**  
[Download from OneDrive](https://1drv.ms/f/c/2d8763f435503032/IgDXzzxzqgXwTKhdc-vmg2UCAcWmzb5pxau_6KJKtIFPxTc?e=EukkVo)

By default, extract or place the downloaded data at:

```text
SST-BR/
├── data/
│   └── RHB_train/
├── configs/
├── scripts/
├── sstbr/
└── tests/
```

The loader expects 246 source-segment directories covering 82 subjects, with
three segments per subject:

```text
data/RHB_train/
├── <subject>_1/
│   ├── <recording>_complex_range_matrix.mat
│   └── vital_dict.npy
├── <subject>_2/
│   ├── <recording>_complex_range_matrix.mat
│   └── vital_dict.npy
└── <subject>_3/
    ├── <recording>_complex_range_matrix.mat
    └── vital_dict.npy
```

Each segment must contain exactly one radar MAT file and one `vital_dict.npy`
PPG file. See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for the required MAT
fields and array organization.

## Configure Dataset Path

The canonical configuration is `configs/rhb_full.yaml`. Its default dataset
location is:

```yaml
dataset:
  rhb_root: data/RHB_train
```

No Python source file needs to be modified. If the dataset is stored elsewhere,
change only `dataset.rhb_root` in `configs/rhb_full.yaml`, for example:

```yaml
# Windows
dataset:
  rhb_root: D:/datasets/RHB_train
```

```yaml
# Linux or macOS
dataset:
  rhb_root: /path/to/RHB_train
```

Run the commands from the repository root so that relative paths resolve as
shown above.

## Run the Full Pipeline

From the repository root, run:

```bash
python scripts/run_full_pipeline.py --config configs/rhb_full.yaml
```

A single command executes all four predefined subject-disjoint folds and
performs raw-data loading, PPG label generation, feature extraction, temporal
fusion, model fitting, inference, 10-s aggregation, and pooled evaluation.

The first run performs complete label and radar-feature extraction. Later runs
may reuse caches that pass version, dataset, configuration, and feature-schema
provenance checks:

```bash
python scripts/run_full_pipeline.py --config configs/rhb_full.yaml --reuse-cache
```

## Outputs

The pipeline creates three ignored runtime directories:

- `outputs/` — predictions, metrics, dataset manifest, and training metadata;
- `models/` — fitted baseline and residual estimators for each fold;
- `cache/` — validated PPG-label and radar-feature caches.

The principal output files are:

| File | Contents |
|---|---|
| `outputs/predictions_10s.csv` | Final pooled 10-s predictions and direct PPG references |
| `outputs/metrics_10s.json` | Per-fold and pooled RMSE, MAE, Pearson r, and sample counts |
| `outputs/predictions_local.csv` | Window-level baseline, residual, and HR predictions |
| `outputs/local_metrics.json` | Metrics at the local 5-s prediction scale |
| `outputs/dataset_manifest.csv` | Validated raw-data inventory |
| `outputs/training_report.json` | Fold sizes, model criteria, timing, and decomposition checks |

For the reported endpoint, inspect `outputs/predictions_10s.csv` and the
`pooled` object in `outputs/metrics_10s.json`.

## Evaluation Protocol

The four YAML fold files define subject-disjoint train, validation, and test
sets. Both model branches are fitted using training subjects only. Test-subject
PPG labels are attached after prediction for evaluation and are not used to
construct radar features or target-subject temporal context.

Each pair of adjacent 5-s local predictions is averaged arithmetically to form
one 10-s prediction. The corresponding 10-s reference HR is estimated directly
from the matching 300-sample PPG segment; it is not the mean of two 5-s labels.

All test predictions from the four folds are concatenated before RMSE, MAE, and
Pearson correlation are computed. The pooled metrics are therefore not an
average of fold-level metrics.

## Reference Result

The manuscript reports the following pooled 10-s result:

| N | RMSE (BPM) | MAE (BPM) | Pearson r |
|---:|---:|---:|---:|
| 720 | 8.2711 | 6.3645 | 0.7628 |

These values are manuscript reference results only. The code reports the
metrics obtained from the current dataset, environment, and configuration and
does not enforce equality with stored reference values.

## Paper-to-Code Mapping

| Paper step | Mathematical role | Main implementation |
|---|---|---|
| Raw data and fold inventory | Build keyed radar/PPG segments and subject-disjoint splits | `sstbr/data/io.py` |
| PPG reference generation | Direct 5-s training labels and independent 10-s references | `sstbr/data/ppg_labels.py` |
| Local windowing | Construct each 5-s radar observation | `sstbr/data/windowing.py` |
| Local radar preprocessing | Construct processed phase and spectral signals | `sstbr/features/preprocessing.py` |
| Spatial representation | Range-bin evidence, energy selection, and dynamic tracking | `sstbr/features/spatial.py`, `sstbr/features/_anchor_kernel.py`, `sstbr/features/_spatial_kernel.py` |
| Spectral representation | Cardiac-band and filterbank spectral evidence | `sstbr/features/spectral.py`, `sstbr/features/_spectral_kernel.py` |
| Local representation | Ordered `x_{s,w}` spatial–spectral feature vector | `sstbr/features/extractor.py` |
| Temporal fusion | `a_s = A({x_{s,w}})` radar-only long-time subject context | `sstbr/temporal/fusion.py` |
| Baseline–residual regression | Subject baseline plus local residual | `sstbr/models/baseline_residual.py` |
| 10-s aggregation | Adjacent local predictions to one 10-s estimate | `sstbr/evaluation/aggregation.py` |
| Pooled evaluation | RMSE, MAE, and Pearson `r` over concatenated test predictions | `sstbr/evaluation/metrics.py` |
| End-to-end orchestration | Raw data to saved predictions and metrics | `sstbr/pipeline.py`, `scripts/run_full_pipeline.py` |

See [`docs/METHOD_OVERVIEW.md`](docs/METHOD_OVERVIEW.md) for a compact data-flow
summary.

## Repository Structure

```text
SST-BR/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── config.example.yaml
├── configs/
│   ├── features/
│   ├── folds/
│   └── rhb_full.yaml
├── docs/
├── scripts/
│   ├── run_full_pipeline.py
│   └── verify_installation.py
├── sstbr/
│   ├── data/
│   ├── features/
│   ├── temporal/
│   ├── models/
│   ├── evaluation/
│   ├── utils/
│   └── pipeline.py
└── tests/
```

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated
SST-BR manuscript when using this implementation. Users of the RHB dataset
should also cite the original Radar-APLANC work.

## License

This repository is released under the MIT License. See `LICENSE` for details.
