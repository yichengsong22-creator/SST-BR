"""Convenient full-release facade over the SST-BR components."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .features.extractor import load_feature_contract
from .features.preprocessing import SignalRuntime, WindowComputationCache
from .features.spatial import SpatialExtractor
from .features.spectral import SpectralExtractor
from .models.baseline_residual import BaselineResidualRegressor
from .temporal.fusion import fuse_one_subject


class SSTBRCore:
    """Full executable SST-BR representation and baseline–residual API."""

    def __init__(self, project_root: str | Path | None = None, radar_fs: float = 120.0) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.radar_fs = float(radar_fs)
        manifest, block_names = load_feature_contract(self.project_root)
        self.feature_names = [item["feature_name"] for item in manifest]
        self.cache = WindowComputationCache()
        runtime = SignalRuntime(self.cache)
        self.spatial = SpatialExtractor(
            runtime,
            Path("."),
            block_names["original"],
            block_names["energy"],
            block_names["tracking"],
        )
        self.spectral = SpectralExtractor(runtime, block_names["filterbank"])

    def extract_local_representation(
        self,
        radar_window: np.ndarray,
        range_grid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract the complete ordered local SST-BR representation."""
        window = np.asarray(radar_window)
        if window.ndim != 2 or window.shape[1] < 32:
            raise ValueError("radar_window must have shape (range_bins, samples).")
        if range_grid is None:
            range_grid = np.linspace(0.0, 2.5, window.shape[0])
        range_grid = np.asarray(range_grid, dtype=float)
        if range_grid.shape != (window.shape[0],):
            raise ValueError("range_grid must match the radar range-bin axis.")
        self.cache.begin_session()
        spatial_values, _ = self.spatial.extract(
            "synthetic_1/0",
            ".",
            0,
            window,
            range_grid,
            window.shape[1],
            self.radar_fs,
            (window, range_grid),
        )
        working = {
            "ids": "synthetic_1/0",
            "subject_id": "synthetic",
            "source_segment_id": ".",
            "local_window_id": 0,
            **spatial_values,
        }
        spectral_values = self.spectral.extract(working, window, self.radar_fs)
        combined = {**working, **spectral_values}
        return np.asarray([combined[name] for name in self.feature_names], dtype=float)

    @staticmethod
    def fuse_temporal_context(window_representations: np.ndarray) -> np.ndarray:
        """Apply the full-release time–feature fusion to one subject sequence."""
        return fuse_one_subject(window_representations)

    def fit(
        self,
        window_representations: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        fold: int = 1,
    ) -> "SSTBRCore":
        """Fit the SST-BR baseline–residual models for an executable example."""
        config = yaml.safe_load((self.project_root / "configs" / "rhb_full.yaml").read_text(encoding="utf-8"))
        criteria = config["models"]["criteria"][f"fold{fold}"]
        self.regressor = BaselineResidualRegressor(
            config["models"]["baseline"],
            config["models"]["residual"],
            criteria["et_criterion"],
            criteria["rf_criterion"],
        ).fit(window_representations, labels, subject_ids)
        return self

    def predict_baseline(self, temporal_context: np.ndarray) -> float:
        """Predict the subject baseline component."""
        return self.regressor.predict_baseline(temporal_context)

    def predict_residual(self, current_window_representation: np.ndarray) -> float:
        """Predict the current-window residual component."""
        return self.regressor.predict_residual(current_window_representation)

    def combine_hr(self, baseline: float, residual: float) -> float:
        """Combine the two fitted branches into a heart-rate estimate."""
        return self.regressor.combine_hr(baseline, residual)
