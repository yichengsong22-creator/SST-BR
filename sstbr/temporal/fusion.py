"""Public temporal data-flow structures for the executable demonstration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalContext:
    """Validated educational context derived from multiple local windows."""

    values: np.ndarray
    window_count: int


def demo_temporal_context(window_representations: np.ndarray) -> TemporalContext:
    """Fuse demo windows with a deliberately simple, non-paper summary."""
    values = np.asarray(window_representations, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("window_representations must be a finite 2-D array with at least two rows.")
    position = np.linspace(-1.0, 1.0, values.shape[0])
    trend = position @ values / float(position @ position)
    context = np.concatenate([values.mean(axis=0), trend, values[-1]])
    return TemporalContext(context, values.shape[0])
