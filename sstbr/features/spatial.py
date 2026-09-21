"""Public spatial-stage contracts and educational summaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialView:
    """Small educational view of how a radar window varies over range."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if np.asarray(self.values).ndim != 1:
            raise ValueError("SpatialView.values must be one-dimensional.")


def demo_spatial_view(radar_window: np.ndarray) -> SpatialView:
    """Return generic magnitude statistics for the executable demonstration.

    This is deliberately not the spatial representation used in the paper.
    """
    window = np.asarray(radar_window)
    if window.ndim != 2 or window.shape[1] == 0:
        raise ValueError("radar_window must have shape (range_bins, samples).")
    magnitude = np.abs(window)
    profile = magnitude.mean(axis=1)
    coordinates = np.linspace(0.0, 1.0, len(profile))
    total = float(profile.sum()) + np.finfo(float).eps
    center = float(np.dot(coordinates, profile) / total)
    spread = float(np.sqrt(np.dot((coordinates - center) ** 2, profile) / total))
    values = np.array([magnitude.mean(), magnitude.std(), profile.max(), center, spread], dtype=float)
    return SpatialView(values)
