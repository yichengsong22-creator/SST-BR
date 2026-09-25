"""Original RHB periodogram HR definition and strict 5-s/10-s labels."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

from .windowing import PPG_FS, TOTAL_LOCAL_WINDOWS, TOTAL_SOURCE_SEGMENTS

BUTTER_ORDER = 6
HR_LOW_BPM = 45.0
HR_HIGH_BPM = 150.0
NFFT = 18000


def estimate_ppg_hr(values: np.ndarray) -> float:
    """Apply the SST-BR 6th-order 0.75--2.5 Hz filter and nfft=18000 periodogram."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"PPG must be finite and 1-D, got {values.shape}")
    b, a = signal.butter(BUTTER_ORDER, [HR_LOW_BPM / 60, HR_HIGH_BPM / 60], btype="bandpass", fs=PPG_FS)
    filtered = signal.filtfilt(b, a, np.double(values))
    frequencies, power = signal.periodogram(filtered, nfft=NFFT, fs=PPG_FS)
    mask = (frequencies >= HR_LOW_BPM / 60) & (frequencies <= HR_HIGH_BPM / 60)
    return float((frequencies * mask)[np.argmax(power * mask)] * 60)


def generate_ppg_labels(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 1476 5-s labels and 738 independently estimated 10-s labels."""
    rows5, rows10 = [], []
    for source in manifest.itertuples(index=False):
        path = Path(source.ppg_file)
        raw = np.asarray(np.load(path, allow_pickle=True))
        if raw.ndim != 1 or len(raw) < 900:
            raise ValueError(f"PPG length < 900 or not 1-D: {path} {raw.shape}")
        values = np.asarray(raw[:900], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite PPG: {path}")
        for local_window_id, start in enumerate(range(0, 900, 150)):
            rows5.append({
                "subject_id": str(source.subject_id), "source_segment_id": source.source_segment_id,
                "local_window_id": local_window_id,
                "global_window_id": (int(source.source_segment_ordinal) - 1) * 6 + local_window_id,
                "start_sample": start, "end_sample": start + 150,
                "hr_bpm": estimate_ppg_hr(values[start:start + 150]), "source_file": str(path),
            })
        for evaluation_window_id, start in enumerate(range(0, 900, 300)):
            rows10.append({
                "subject_id": str(source.subject_id), "source_segment_id": source.source_segment_id,
                "evaluation_window_id": evaluation_window_id,
                "start_sample": start, "end_sample": start + 300,
                "hr_bpm": estimate_ppg_hr(values[start:start + 300]), "source_file": str(path),
            })
    labels5, labels10 = pd.DataFrame(rows5), pd.DataFrame(rows10)
    if len(labels5) != TOTAL_LOCAL_WINDOWS or len(labels10) != TOTAL_SOURCE_SEGMENTS * 3:
        raise ValueError("Unexpected PPG output cardinality")
    if labels5.duplicated(["subject_id", "source_segment_id", "local_window_id"]).any() or labels10.duplicated(["subject_id", "source_segment_id", "evaluation_window_id"]).any():
        raise ValueError("Duplicate PPG label key")
    return labels5, labels10
