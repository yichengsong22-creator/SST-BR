"""Canonical SST-BR spatial feature implementation.

This facade combines anchor-centered, energy-supported, and dynamic tracking
features and is the entry point used by the release pipeline.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .preprocessing import SignalRuntime
from . import _anchor_kernel as anchor
from . import _spatial_kernel as refinement


class SpatialExtractor:
    """Extract Original/Energy/Tracking dictionaries for one 5-s window."""

    def __init__(self, runtime: SignalRuntime, dataset_root: Path, original_names: list[str], energy_names: list[str], tracking_names: list[str]):
        self.runtime = runtime
        self.dataset_root = dataset_root
        self.original_names = original_names
        self.energy_names = energy_names
        self.tracking_names = tracking_names
        self.anchor = anchor.OriginalExtractor(runtime)
        self.refinement = refinement.EnergyTrackingExtractor(runtime)

    def extract(self, ids: str, source_segment_id: str, local_window_id: int, window: np.ndarray, range_grid: np.ndarray, full_matrix_length: int, fs: float, loaded: tuple[np.ndarray, np.ndarray]) -> tuple[dict[str, float], dict]:
        """Return spatial features and audit metadata; labels are never accepted."""
        anchor.DATA_ROOT = self.dataset_root
        anchor.load_radar_mat = lambda _directory: loaded
        # Legacy adapter fields are placeholders and are excluded from the
        # final SST-BR feature manifest.
        values = anchor.extract_features_for_window(ids, source_segment_id, local_window_id, local_window_id * 5, (local_window_id + 1) * 5, 0, "feature_only")
        original = {name: values[name] for name in self.original_names}
        selection = refinement.select_energy_bins(window, range_grid)
        energy = {
            "erg_radar_fs": float(fs), "erg_n_range_bins": int(window.shape[0]),
            "erg_n_time_points_trial": int(full_matrix_length), "erg_n_time_points_window": int(window.shape[1]),
        }
        original_series = pd.Series(original)
        energy.update(refinement.energy_distribution_features(selection, original_series))
        energy.update(refinement.extract_erg_spectral_features(window, selection["selected_bins"], selection["energy_norm_all"], fs))
        tracking = refinement.extract_rt_features_from_energy_bins(window, selection["selected_bins"], selection["energy_norm_all"], fs, original_series)
        state = self.refinement.last_tracking_path or {}
        path = refinement.dp_track_bins(state.get("scores", np.empty((0, 0))), state.get("candidate_bins", np.array([], int))) if state else []
        combined = {**original, **{name: energy[name] for name in self.energy_names}, **{name: tracking[name] for name in self.tracking_names}}
        audit = {"selected_bin": int(original["selected_bin"]), "tracking_path": [int(x) for x in path], "energy_selected_bins": [int(x) for x in selection["selected_bins"]]}
        return combined, audit
