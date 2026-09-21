"""Public spectral-stage contracts and educational summaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralView:
    """Small educational view of slow-time frequency content."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if np.asarray(self.values).ndim != 1:
            raise ValueError("SpectralView.values must be one-dimensional.")


def demo_spectral_view(radar_window: np.ndarray) -> SpectralView:
    """Return generic Fourier summaries for the executable demonstration.

    No paper filterbank, candidate rule, or quality calculation is present.
    """
    window = np.asarray(radar_window)
    if window.ndim != 2 or window.shape[1] < 4:
        raise ValueError("radar_window must have shape (range_bins, samples).")
    trace = np.unwrap(np.angle(window.mean(axis=0)))
    spectrum = np.abs(np.fft.rfft(trace - trace.mean()))
    frequencies = np.fft.rfftfreq(len(trace), d=1.0)
    non_dc = spectrum[1:]
    peak_index = int(np.argmax(non_dc)) + 1
    total = float(spectrum.sum()) + np.finfo(float).eps
    values = np.array([
        spectrum.mean(),
        spectrum.std(),
        frequencies[peak_index],
        float(np.dot(frequencies, spectrum) / total),
    ])
    return SpectralView(values)
