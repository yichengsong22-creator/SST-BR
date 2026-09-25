"""Multi-band K=3/4 filterbank spectral block (33 retained dimensions)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import SignalRuntime
from ..data.windowing import crop_radar_window
from . import _spectral_kernel as kernel


class SpectralExtractor:
    """Extract the current-window filterbank features."""

    def __init__(self, runtime: SignalRuntime, all_names: list[str]):
        self.all_names = all_names
        self.kernel = kernel.FilterbankExtractor(runtime)

    def extract(self, spatial_row: dict, matrix: np.ndarray, fs: float) -> dict[str, float]:
        """Return the 40-D internal filterbank dictionary for selection to 33-D."""
        window, _, _ = crop_radar_window(matrix, int(spatial_row["local_window_id"]))
        values = kernel.process_one_row(pd.Series(spatial_row), matrix_10s=window, fs=fs)
        values.update({"avmd_radar_fs": fs, "avmd_n_time_points_window": window.shape[1], "avmd_n_range_bins": matrix.shape[0]})
        return {name: values[name] for name in self.all_names}
