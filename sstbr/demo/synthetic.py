"""Synthetic signals and a lightweight full-source SST-BR demonstration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import SSTBRCore
from ..data.ppg_labels import estimate_ppg_hr
from ..evaluation import aggregate_consecutive_predictions, attach_direct_references, regression_metrics


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


def run_demo(seed: int = 2026) -> dict:
    """Exercise the full representation, fusion, regression, and evaluation code."""
    recording = generate_synthetic_recording(seed)
    core = SSTBRCore(radar_fs=recording.radar_hz)
    representations = np.vstack([
        core.extract_local_representation(window, recording.range_grid)
        for window in recording.radar_windows
    ])
    representations = np.nan_to_num(representations, nan=0.0, posinf=0.0, neginf=0.0)
    local_labels = np.asarray([
        estimate_ppg_hr(recording.ppg[index * 150:(index + 1) * 150])
        for index in range(6)
    ])
    subject_offsets = np.linspace(-12.0, 12.0, 12)
    training_features = np.vstack([
        representations + offset * 0.002 for offset in subject_offsets
    ])
    training_labels = np.concatenate([local_labels + offset for offset in subject_offsets])
    training_subjects = np.repeat([f"synthetic-{index}" for index in range(12)], 6)
    core.fit(training_features, training_labels, training_subjects)
    context = core.fuse_temporal_context(representations)
    baseline = core.predict_baseline(context)
    residuals = np.asarray([core.predict_residual(row) for row in representations])
    local_predictions = np.asarray([core.combine_hr(baseline, value) for value in residuals])
    prediction_table = pd.DataFrame({
        "fold": ["synthetic"] * 6,
        "subject_id": ["synthetic"] * 6,
        "source_segment_id": ["synthetic-segment"] * 6,
        "local_window_id": range(6),
        "pred_hr": local_predictions,
    })
    direct = pd.DataFrame({
        "subject_id": ["synthetic"] * 3,
        "source_segment_id": ["synthetic-segment"] * 3,
        "evaluation_window_id": range(3),
        "hr_bpm": [
            estimate_ppg_hr(recording.ppg[index * 300:(index + 1) * 300])
            for index in range(3)
        ],
    })
    evaluation = attach_direct_references(aggregate_consecutive_predictions(prediction_table), direct)
    return {
        "representation_shape": representations.shape,
        "context_shape": context.shape,
        "baseline": baseline,
        "residual": residuals[-1],
        "final": local_predictions[-1],
        "local_windows": len(local_predictions),
        "evaluation_samples": len(evaluation),
        "metrics": regression_metrics(evaluation.gt_hr, evaluation.pred_hr),
    }
