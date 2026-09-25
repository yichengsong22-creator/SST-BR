"""All-18, feature-wise subject context construction."""
from __future__ import annotations

from functools import lru_cache
import warnings
import numpy as np
import pandas as pd

STATISTICS = ("mean", "median", "std", "min", "max", "range")


def _mean(values: np.ndarray) -> np.ndarray:
    total = np.zeros(values.shape[1])
    compensation = np.zeros(values.shape[1])
    count = np.zeros(values.shape[1], dtype=int)
    for row in values:
        valid = np.isfinite(row)
        y = np.where(valid, row - compensation, 0)
        updated = total + y
        compensation = np.where(valid, (updated - total) - y, compensation)
        total = np.where(valid, updated, total)
        count += valid
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _sample_std(values: np.ndarray) -> np.ndarray:
    count = np.zeros(values.shape[1], dtype=int)
    mean = np.zeros(values.shape[1])
    m2 = np.zeros(values.shape[1])
    for row in values:
        valid = np.isfinite(row)
        next_count = count + valid
        delta = row - mean
        next_mean = np.where(valid, mean + delta / np.where(next_count > 0, next_count, 1), mean)
        m2 = np.where(valid, m2 + delta * (row - next_mean), m2)
        mean, count = next_mean, next_count
    return np.sqrt(np.divide(m2, count - 1, out=np.full_like(m2, np.nan), where=count > 1))


@lru_cache(maxsize=2)
def context_feature_names(features: tuple[str, ...]) -> list[str]:
    """Preserve the interleaved SST-BR statistic order plus trailing ranges."""
    first = [f"subj_{feature}__{stat}" for feature in features for stat in STATISTICS[:-1]]
    return ["n_windows", *first, *[f"subj_{feature}__range" for feature in features]]


def fuse_one_subject(window_features: np.ndarray) -> np.ndarray:
    """Fuse an arbitrary non-empty sequence with SST-BR temporal statistics."""
    values = np.asarray(window_features, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("window_features must have shape (windows, features).")
    # Paper notation: a_s = A({x_{s,w}}), where A summarizes all local radar
    # representations for subject s into a radar-only temporal context.
    mean = _mean(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        minimum = np.nanmin(values, axis=0)
        maximum = np.nanmax(values, axis=0)
    std = _sample_std(values)
    interleaved = np.column_stack((mean, median, std, minimum, maximum)).reshape(-1)
    return np.concatenate(([float(values.shape[0])], interleaved, maximum - minimum))


def build_temporal_context(window_features: np.ndarray, metadata: pd.DataFrame, feature_names: list[str], split_by_subject: dict[str, str]) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Aggregate exactly 18 x 331 radar features to one 1987-D row per subject."""
    columns = context_feature_names(tuple(feature_names))
    subjects = metadata.subject_id.astype(str).to_numpy()
    ordered = sorted(np.unique(subjects))
    rows, meta_rows = [], []
    for subject in ordered:
        values = window_features[subjects == subject]
        if values.shape != (18, 331):
            raise ValueError(f"Unexpected temporal input shape for {subject}: {values.shape}")
        rows.append(fuse_one_subject(values))
        meta_rows.append({"subject_id": subject, "split": split_by_subject[subject]})
    context = np.vstack(rows)
    if context.shape != (82, 1987) or len(columns) != 1987:
        raise ValueError(f"Unexpected temporal context shape: {context.shape}")
    return context, pd.DataFrame(meta_rows), columns
