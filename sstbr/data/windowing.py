"""Authoritative neutral recording/segment/window indexing rules."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

RADAR_FS = 120.0
PPG_FS = 30.0
SOURCE_SEGMENT_SECONDS = 30.0
LOCAL_WINDOW_SECONDS = 5.0
LOCAL_WINDOWS_PER_SEGMENT = 6
SOURCE_SEGMENTS_PER_SUBJECT = 3
LOCAL_WINDOWS_PER_SUBJECT = 18
TOTAL_SUBJECTS = 82
TOTAL_SOURCE_SEGMENTS = 246
TOTAL_LOCAL_WINDOWS = 1476


@dataclass(frozen=True)
class WindowBounds:
    local_window_id: int
    radar_start: int
    radar_end_nominal: int
    ppg_start: int
    ppg_end: int


def local_window_bounds(local_window_id: int) -> WindowBounds:
    """Return zero-based 5-s radar/PPG sample bounds."""
    if local_window_id not in range(LOCAL_WINDOWS_PER_SEGMENT):
        raise ValueError(f"local_window_id must be 0..5, got {local_window_id}")
    radar_width = int(RADAR_FS * LOCAL_WINDOW_SECONDS)
    ppg_width = int(PPG_FS * LOCAL_WINDOW_SECONDS)
    return WindowBounds(
        local_window_id,
        local_window_id * radar_width,
        (local_window_id + 1) * radar_width,
        local_window_id * ppg_width,
        (local_window_id + 1) * ppg_width,
    )


def crop_radar_window(matrix: np.ndarray, local_window_id: int) -> tuple[np.ndarray, int, int]:
    """Crop without interpolation/padding; the final window may contain 598--600 samples."""
    bounds = local_window_bounds(local_window_id)
    if matrix.ndim != 2 or matrix.shape[1] <= bounds.radar_start:
        raise ValueError(f"Cannot crop window {local_window_id} from radar shape {matrix.shape}")
    end = min(bounds.radar_end_nominal, matrix.shape[1])
    return matrix[:, bounds.radar_start:end], bounds.radar_start, end


def tracking_edges(actual_length: int) -> np.ndarray:
    """Five approximately equal short segments used inside Spatial tracking."""
    return np.linspace(0, actual_length, 6).round().astype(int)
