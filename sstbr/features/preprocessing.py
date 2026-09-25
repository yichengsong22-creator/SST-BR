"""Shared exact-equivalent numerical runtime for one subject/session."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal


def array_key(array: np.ndarray) -> tuple:
    value = np.asarray(array)
    return (int(value.__array_interface__["data"][0]), tuple(value.shape), tuple(value.strides), value.dtype.str)


@dataclass
class WindowComputationCache:
    """Strict intermediate cache; cleared at each subject/session boundary."""
    values: dict = field(default_factory=dict)
    owners: list = field(default_factory=list)

    def begin_session(self) -> None:
        self.values.clear(); self.owners.clear()

    def get_or_compute(self, namespace: str, key: tuple, compute: Callable, owners: tuple = ()):
        composite = (namespace, key)
        if composite not in self.values:
            self.values[composite] = compute(); self.owners.extend(owners)
        return self.values[composite]

    def store(self, namespace: str, key: tuple, value, owners: tuple = ()) -> None:
        self.values[(namespace, key)] = value; self.owners.extend(owners)


@dataclass
class SignalRuntime:
    """Precomputed filters and immutable frequency metadata."""
    cache: WindowComputationCache
    sos: dict = field(default_factory=dict)
    frequency: dict = field(default_factory=dict)

    def butter_sos(self, order: int, fs: float, low: float, high: float) -> np.ndarray:
        key = (int(order), float(fs), float(low), float(high))
        if key not in self.sos:
            nyquist = 0.5 * fs
            self.sos[key] = signal.butter(order, [low/nyquist, high/nyquist], btype="bandpass", output="sos")
        return self.sos[key]

    def frequency_grid(self, fs: float, nfft: int) -> np.ndarray:
        key = (float(fs), int(nfft))
        if key not in self.frequency:
            self.frequency[key] = np.fft.rfftfreq(nfft, 1.0/fs)
        return self.frequency[key]


def batch_unwrap_scalar_detrend(matrix: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Batch unwrap, then scalar detrend (batch detrend failed the 1e-12 gate)."""
    phase = np.nan_to_num(np.asarray(np.unwrap(np.angle(matrix[np.asarray(bins, int)]), axis=-1), float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.vstack([signal.detrend(row) for row in phase])


def batch_filter(values: np.ndarray, sos: np.ndarray) -> np.ndarray:
    try:
        result = signal.sosfiltfilt(sos, values, axis=-1)
    except Exception:
        result = signal.sosfilt(sos, values, axis=-1)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def batch_welch(values: np.ndarray, fs: float, nperseg: int, noverlap: int, nfft: int, scaling: str = "density"):
    return signal.welch(values, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                        nfft=nfft, detrend="constant", scaling=scaling, axis=-1)
