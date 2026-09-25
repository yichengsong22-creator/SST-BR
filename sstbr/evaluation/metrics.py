"""Metric definitions for concatenated 10-s samples."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def regression_metrics(ground_truth, prediction) -> dict[str, float | int]:
    """Return sample count, RMSE, MAE and Pearson r."""
    ground_truth = np.asarray(ground_truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(ground_truth) == 0 or not (np.isfinite(ground_truth).all() and np.isfinite(prediction).all()):
        raise ValueError("Metrics require non-empty finite vectors")
    error = prediction - ground_truth
    return {"N": int(len(error)), "RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(np.abs(error))), "r": float(pearsonr(ground_truth, prediction).statistic)}
