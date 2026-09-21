"""Public, non-overlapping local-window indexing utilities."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    """Zero-based sample and time bounds for one local window."""

    window_id: int
    start_seconds: float
    end_seconds: float
    start_sample: int
    end_sample_nominal: int


def make_window_specs(segment_seconds: float, window_seconds: float, sampling_rate: float) -> list[WindowSpec]:
    """Create non-overlapping windows that exactly tile a nominal segment."""
    if min(segment_seconds, window_seconds, sampling_rate) <= 0:
        raise ValueError("Durations and sampling rate must be positive.")
    count_float = segment_seconds / window_seconds
    count = int(round(count_float))
    if not np.isclose(count_float, count):
        raise ValueError("window_seconds must exactly tile segment_seconds.")
    width_float = window_seconds * sampling_rate
    width = int(round(width_float))
    if not np.isclose(width_float, width):
        raise ValueError("Window duration must map to an integer sample count.")
    return [WindowSpec(index, index * window_seconds, (index + 1) * window_seconds, index * width, (index + 1) * width) for index in range(count)]


def crop_available_window(values: np.ndarray, spec: WindowSpec, sample_axis: int = -1) -> np.ndarray:
    """Crop available samples without padding, interpolation, or resampling."""
    values = np.asarray(values)
    axis = sample_axis % values.ndim
    available = values.shape[axis]
    if available <= spec.start_sample:
        raise ValueError(f"Window {spec.window_id} starts beyond the available signal.")
    end = min(spec.end_sample_nominal, available)
    index = [slice(None)] * values.ndim
    index[axis] = slice(spec.start_sample, end)
    return values[tuple(index)]
