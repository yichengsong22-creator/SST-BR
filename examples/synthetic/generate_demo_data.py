"""Deterministic synthetic radar-like and PPG signals for Binder examples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticRecording:
    """Small in-memory recording with six local windows and direct PPG."""

    radar_windows: np.ndarray
    range_grid: np.ndarray
    ppg: np.ndarray
    radar_hz: float
    ppg_hz: float


def generate_synthetic_recording(seed: int = 2026) -> SyntheticRecording:
    """Create repeatable educational signals; no real subject data are used."""
    rng = np.random.default_rng(seed)
    radar_hz = 120.0
    ppg_hz = 30.0
    local_seconds = 5.0
    range_bins = 12
    local_rates = np.array([58.0, 62.0, 78.0, 84.0, 98.0, 104.0])
    range_grid = np.linspace(0.25, 2.25, range_bins)
    radar_windows = []
    for index, rate in enumerate(local_rates):
        time = np.arange(int(local_seconds * radar_hz)) / radar_hz
        target = 4 + (index % 3)
        envelope = np.exp(-0.5 * ((np.arange(range_bins) - target) / 1.3) ** 2)[:, None]
        phase = 0.55 * np.sin(2.0 * np.pi * (rate / 60.0) * time)
        signal = envelope * np.exp(1j * phase)[None, :]
        noise = 0.08 * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
        radar_windows.append(signal + noise)
    ppg_parts = []
    for rate in (60.0, 81.0, 101.0):
        time = np.arange(int(10.0 * ppg_hz)) / ppg_hz
        fundamental = np.sin(2.0 * np.pi * (rate / 60.0) * time)
        harmonic = 0.18 * np.sin(4.0 * np.pi * (rate / 60.0) * time + 0.4)
        ppg_parts.append(fundamental + harmonic + 0.015 * rng.normal(size=len(time)))
    return SyntheticRecording(np.stack(radar_windows), range_grid, np.concatenate(ppg_parts), radar_hz, ppg_hz)
