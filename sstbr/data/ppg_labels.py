"""Public PPG heart-rate labels for supervision and evaluation."""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import signal

from .io import RawSegment, load_ppg_signal
from .windowing import make_window_specs


def estimate_ppg_hr(
    values: np.ndarray,
    sampling_rate: float = 30.0,
    lower_bpm: float = 45.0,
    upper_bpm: float = 150.0,
    filter_order: int = 6,
    frequency_resolution_bpm: float = 0.1,
) -> float:
    """Estimate HR from filtered PPG using the dominant periodogram peak."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("PPG input must be a non-empty finite one-dimensional array.")
    coefficients = signal.butter(
        filter_order,
        [lower_bpm / 60.0, upper_bpm / 60.0],
        btype="bandpass",
        fs=sampling_rate,
    )
    filtered = signal.filtfilt(*coefficients, values)
    nfft = int(round(60.0 * sampling_rate / frequency_resolution_bpm))
    frequencies, power = signal.periodogram(filtered, nfft=nfft, fs=sampling_rate)
    allowed = (frequencies >= lower_bpm / 60.0) & (frequencies <= upper_bpm / 60.0)
    if not np.any(allowed):
        raise ValueError("No periodogram bins fall inside the requested HR range.")
    masked_power = np.where(allowed, power, 0.0)
    return float(frequencies[int(np.argmax(masked_power))] * 60.0)


def generate_ppg_label_tables(
    segments: Iterable[RawSegment],
    sampling_rate: float = 30.0,
    segment_seconds: float = 30.0,
    local_window_seconds: float = 5.0,
    evaluation_window_seconds: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate local supervision labels and independently estimated evaluation labels."""
    local_specs = make_window_specs(segment_seconds, local_window_seconds, sampling_rate)
    evaluation_specs = make_window_specs(segment_seconds, evaluation_window_seconds, sampling_rate)
    required_samples = int(round(segment_seconds * sampling_rate))
    local_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    for segment in segments:
        ppg = load_ppg_signal(segment.ppg_path, required_samples)[:required_samples]
        for spec in local_specs:
            local_rows.append({
                "subject_id": segment.subject_id,
                "source_segment_id": segment.source_segment_id,
                "local_window_id": spec.window_id,
                "start_sample": spec.start_sample,
                "end_sample": spec.end_sample_nominal,
                "hr_bpm": estimate_ppg_hr(ppg[spec.start_sample:spec.end_sample_nominal], sampling_rate),
            })
        for spec in evaluation_specs:
            evaluation_rows.append({
                "subject_id": segment.subject_id,
                "source_segment_id": segment.source_segment_id,
                "evaluation_window_id": spec.window_id,
                "start_sample": spec.start_sample,
                "end_sample": spec.end_sample_nominal,
                "hr_bpm": estimate_ppg_hr(ppg[spec.start_sample:spec.end_sample_nominal], sampling_rate),
            })
    local = pd.DataFrame(local_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    if local.empty or evaluation.empty:
        raise ValueError("No PPG labels were generated.")
    if local.duplicated(["subject_id", "source_segment_id", "local_window_id"]).any():
        raise ValueError("Duplicate local-label key.")
    if evaluation.duplicated(["subject_id", "source_segment_id", "evaluation_window_id"]).any():
        raise ValueError("Duplicate evaluation-label key.")
    return local, evaluation
