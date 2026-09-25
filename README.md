# SST-BR

**Spatial–Spectral–Temporal Baseline–Residual Learning for mmWave Radar Heart-Rate Estimation**

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TODO-BEFORE-PUBLIC-UPLOAD/sstbr/main?labpath=notebooks%2F00_quick_start.ipynb)

**Full executable release for reproducing the SST-BR pipeline and evaluation.**

## Overview

This repository contains the complete SST-BR implementation used for mmWave radar heart-rate estimation. The full pipeline includes:

- raw radar and PPG data loading;
- PPG-based reference generation;
- local radar window construction;
- spatial and spectral feature extraction;
- long-time temporal feature fusion;
- baseline–residual regression;
- four-fold subject-disjoint evaluation; and
- direct 10-second HR evaluation with pooled RMSE, MAE, and Pearson correlation.

For reviewers who want to reproduce the reported experiment, **the full pipeline described below is the primary entry point**. The synthetic demo and Binder notebooks are optional utilities for lightweight inspection only.

---

## 1. Full Pipeline Reproduction

### 1.1 Requirements

The release has been validated with Python 3.11. Dependency versions are listed in:

- `requirements.txt`
- `binder/environment.yml`

From the repository root, create and install the environment:

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.

python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Verify the installation:

```bash
python scripts/verify_installation.py
```

A successful installation prints:

```text
Installation verified successfully.
```

### 1.2 Download the RHB Dataset

The experiments in this repository use the **RHB dataset introduced and released with Radar-APLANC** by Wang et al. [1]. The original dataset and its accompanying code were made publicly available through the official Radar-APLANC repository.

For reviewer convenience, the RHB data package used to run the full SST-BR pipeline locally can be downloaded here:

**RHB dataset download:**
https://1drv.ms/f/c/2d8763f435503032/IgDXzzxzqgXwTKhdc-vmg2UCAcWmzb5pxau_6KJKtIFPxTc?e=EukkVo

After downloading, place the dataset at:

```text
data/RHB_train
```

Alternatively, keep the dataset at another local location and update `dataset.rhb_root` in `configs/rhb_full.yaml`.

The RHB dataset is credited to its original authors. If you use the dataset, please cite the Radar-APLANC paper:

> Y. Wang, Z. Sun, X. Cheng, Z. He, and X. Li, “Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 40, no. 12, pp. 10270–10278, 2026, doi: 10.1609/aaai.v40i12.37996.


### 1.3 Run the Complete SST-BR Pipeline

From the repository root, run:

```bash
python scripts/run_full_pipeline.py --config configs/rhb_full.yaml
```

This command executes the complete raw-data-to-result pipeline, including:

```text
RHB radar / PPG data
        ↓
dataset manifest and PPG reference generation
        ↓
local radar window construction
        ↓
spatial–spectral feature extraction
        ↓
long-time temporal fusion
        ↓
baseline–residual model training and inference
        ↓
10-s prediction aggregation
        ↓
four-fold pooled evaluation
```

The evaluation is subject-disjoint across folds. Test-subject PPG labels are used only after prediction for evaluation; they are not used to construct radar features or target-subject temporal context.

### 1.4 Reusing Validated Feature Caches

After a successful initial run, validated local caches can be reused with:

```bash
python scripts/run_full_pipeline.py --config configs/rhb_full.yaml --reuse-cache
```

Cache reuse is protected by configuration, dataset, version, and feature-schema provenance checks.

### 1.5 Generated Outputs

The full pipeline generates its runtime artifacts locally.

Main evaluation outputs are written to:

```text
outputs/
```

including prediction tables, metrics, dataset metadata, and training metadata.

Fitted fold models are written to:

```text
models/
```

Feature and label caches are written under:

```text
cache/
```

These directories are generated at runtime and are excluded from version control.

---

## 2. Evaluation Protocol

SST-BR operates on local radar observations and evaluates the final output at a fixed 10-second scale.

Two consecutive local predictions are averaged to obtain one 10-second HR prediction. The corresponding reference HR is estimated **directly from the full 10-second PPG segment** rather than by averaging shorter PPG labels.

For the final reported result:

1. each fold uses subject-disjoint train, validation, and test sets;
2. predictions from all four test folds are concatenated; and
3. pooled RMSE, MAE, and Pearson correlation are computed once over the complete pooled test set.

This avoids reporting a simple arithmetic average of fold-level metrics.

### Reference Result Reported in the Manuscript

The code always reports the metrics computed from the dataset, software environment, and configuration used for the current run. The values below are provided only as manuscript reference results and are **not enforced by the software**.

| N | RMSE (BPM) | MAE (BPM) | Pearson r |
|---:|---:|---:|---:|
| 720 | 8.2711 | 6.3645 | 0.7628 |

---

## 3. Method Summary

SST-BR separates short-time radar evidence from longer-time subject context.

For each local radar window, the pipeline extracts a structured spatial–spectral representation. Local representations from the same subject are then fused into a radar-only long-time temporal context. This shared context is used by the baseline branch, while the current local representation is retained by the residual branch for window-specific correction.

The final local HR estimate is obtained from the baseline and residual components and is subsequently aggregated to the 10-second evaluation scale.

Additional implementation details are available in:

- [`docs/METHOD_OVERVIEW.md`](docs/METHOD_OVERVIEW.md)
- `sstbr/features/`
- `sstbr/temporal/`
- `sstbr/models/`

---

## 4. Repository Structure

```text
sstbr/
├─ data/          # raw data adapters, windowing, and PPG reference generation
├─ features/      # spatial and spectral radar feature extraction
├─ temporal/      # long-time temporal feature fusion
├─ models/        # baseline–residual regression
├─ evaluation/    # 10-s aggregation and evaluation metrics
├─ demo/          # optional synthetic demonstration helpers
└─ utils/         # configuration and file utilities
```

Other important directories:

- `configs/` — full-pipeline configuration, fold definitions, and feature specifications
- `scripts/` — full-pipeline entry point, installation verification, and optional demo
- `notebooks/` — reviewer-oriented interactive walkthroughs
- `tests/` — software and scientific-invariant tests
- `docs/` — method and environment documentation

---

## 5. Optional Synthetic Demo

The synthetic demo is **not required for reproducing the manuscript result**. It is provided only as a lightweight way to verify that the installed package and core API run correctly without downloading the RHB dataset.

Run:

```bash
python scripts/run_demo.py
```

The demo uses deterministic synthetic input and does not report an RHB experimental result.

---

## 6. Optional Binder Notebooks

Binder is intended for lightweight method inspection and synthetic examples. It does **not** contain the complete RHB dataset and does not execute the full four-fold training pipeline.

After the final GitHub repository URL is available, the Binder badge at the top of this README should point to:

```text
notebooks/00_quick_start.ipynb
```

For manuscript reproduction, use the local full-pipeline procedure in **Section 1**.

---

## 7. Reproducibility Notes

The public release intentionally keeps runtime validation that protects scientific consistency, including checks for:

- dataset structure and file availability;
- fold integrity and subject disjointness;
- radar and PPG alignment;
- feature ordering and dimensionality;
- aggregation cardinality;
- configuration and cache provenance; and
- finite metric inputs.

The software does **not** require the computed metrics to equal pre-stored manuscript values.

Generated caches, fitted models, predictions, and raw RHB data are excluded from version control.

---

## 8. Citation

Citation metadata are provided in:

```text
CITATION.cff
```

Please cite the associated SST-BR manuscript when using this implementation.

---

## 9. License

This repository is released under the MIT License. See:

```text
LICENSE
```

for details.
