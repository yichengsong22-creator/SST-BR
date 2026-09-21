"""Deterministic educational core used only by the partial-release Binder."""
from __future__ import annotations

import numpy as np

from .features.extractor import WindowRepresentation
from .features.spatial import demo_spatial_view
from .features.spectral import demo_spectral_view
from .models.baseline_residual import combine_hr
from .temporal.fusion import TemporalContext, demo_temporal_context


class DemoCore:
    """Illustrate SST-BR interfaces without reproducing the paper core.

    The statistics and fixed display mappings below are intentionally simple,
    educational, and unrelated to the reported-result implementation.
    """

    def extract_local_representation(self, radar_window: np.ndarray) -> WindowRepresentation:
        """Build a compact demo vector from generic spatial/spectral views."""
        spatial = demo_spatial_view(radar_window)
        spectral = demo_spectral_view(radar_window)
        return WindowRepresentation(np.concatenate([spatial.values, spectral.values]), source="DemoCore")

    def fuse_temporal_context(self, window_representations: np.ndarray) -> TemporalContext:
        """Organize several demo window vectors into one temporal context."""
        return demo_temporal_context(window_representations)

    def predict_baseline(self, temporal_context: TemporalContext) -> float:
        """Return a deterministic display baseline from normalized context."""
        centered = np.tanh(float(np.mean(temporal_context.values)))
        return 74.0 + 4.0 * centered

    def predict_residual(self, current_window_representation: WindowRepresentation) -> float:
        """Return a deterministic display residual from the current window."""
        centered = np.tanh(float(np.std(current_window_representation.values)) - 1.0)
        return 3.0 * centered

    @staticmethod
    def combine_hr(baseline: float, residual: float) -> float:
        """Combine the educational branch estimates."""
        return combine_hr(baseline, residual)
