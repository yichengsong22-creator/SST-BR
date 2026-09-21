"""Public regression metrics and true pooled-fold evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def regression_metrics(ground_truth, predictions) -> dict[str, float | int]:
    """Compute N, RMSE, MAE and Pearson r from finite paired vectors."""
    ground_truth = np.asarray(ground_truth, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if ground_truth.ndim != 1 or predictions.ndim != 1 or ground_truth.shape != predictions.shape:
        raise ValueError("Ground truth and predictions must be one-dimensional vectors of equal length.")
    if len(ground_truth) < 2 or not (np.isfinite(ground_truth).all() and np.isfinite(predictions).all()):
        raise ValueError("Metrics require at least two finite paired samples.")
    error = predictions - ground_truth
    return {
        "N": int(len(error)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "r": float(pearsonr(ground_truth, predictions).statistic),
    }


def fold_and_pooled_metrics(frame: pd.DataFrame, fold_column: str = "fold") -> dict:
    """Compute each fold and then recompute pooled metrics from concatenated rows."""
    required = {fold_column, "gt_hr", "pred_hr"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation table is missing columns: {sorted(missing)}")
    per_fold = {
        str(fold): regression_metrics(part.gt_hr, part.pred_hr)
        for fold, part in frame.groupby(fold_column, sort=False)
    }
    return {"per_fold": per_fold, "pooled": regression_metrics(frame.gt_hr, frame.pred_hr)}
