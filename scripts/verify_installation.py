"""Verify imports and static contracts required by the full RHB pipeline."""
from __future__ import annotations

from pathlib import Path

import yaml

import sstbr
from sstbr.data.io import read_fold
from sstbr.evaluation.aggregation import aggregate_consecutive_predictions
from sstbr.evaluation.metrics import regression_metrics
from sstbr.features.extractor import load_feature_contract
from sstbr.models.baseline_residual import train_and_predict
from sstbr.pipeline import run_full_pipeline
from sstbr.temporal.fusion import build_temporal_context


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "rhb_full.yaml").read_text(encoding="utf-8"))
    required_sections = {"dataset", "models", "evaluation", "paths"}
    if not required_sections.issubset(config) or "rhb_root" not in config["dataset"]:
        raise RuntimeError("The full-pipeline configuration is incomplete.")

    manifest, blocks = load_feature_contract(root)
    if len(manifest) != 331 or sum(map(len, blocks.values())) < 331:
        raise RuntimeError("The SST-BR feature contract is unavailable or incomplete.")

    for index in range(1, 5):
        fold = read_fold(root / "configs" / "folds" / f"fold{index}.yaml")
        sets = {name: set(values) for name, values in fold.items()}
        if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
            raise RuntimeError(f"Fold {index} contains overlapping source segments.")

    required_callables = (
        run_full_pipeline,
        train_and_predict,
        build_temporal_context,
        aggregate_consecutive_predictions,
        regression_metrics,
    )
    if not all(callable(item) for item in required_callables):
        raise RuntimeError("A required full-pipeline entry point is unavailable.")

    print(f"SST-BR {sstbr.__version__}")
    print("Installation verified successfully.")


if __name__ == "__main__":
    main()
