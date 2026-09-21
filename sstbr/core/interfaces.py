"""Stable API reserved for the post-publication implementation."""
from __future__ import annotations

import numpy as np

from .unavailable import CoreImplementationUnavailable


class SSTBRCore:
    """Shape-level exact-core interface; Binder users should use `DemoCore`."""

    def extract_local_representation(self, radar_window: np.ndarray):
        """Return the exact local representation in the future full release."""
        raise CoreImplementationUnavailable()

    def fuse_temporal_context(self, window_representations: np.ndarray):
        """Return the exact fused context in the future full release."""
        raise CoreImplementationUnavailable()

    def predict_baseline(self, temporal_context) -> float:
        """Return the exact baseline branch output in the future full release."""
        raise CoreImplementationUnavailable()

    def predict_residual(self, current_window_representation) -> float:
        """Return the exact residual branch output in the future full release."""
        raise CoreImplementationUnavailable()

    def combine_hr(self, baseline: float, residual: float) -> float:
        """Return the exact final paper prediction in the future full release."""
        raise CoreImplementationUnavailable()
