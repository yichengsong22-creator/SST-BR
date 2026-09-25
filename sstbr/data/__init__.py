"""Raw radar/PPG loading, windowing, and reference-label generation."""

from .io import build_dataset_manifest, load_radar_segment
from .ppg_labels import estimate_ppg_hr, generate_ppg_labels
from .windowing import WindowBounds, crop_radar_window, local_window_bounds

__all__ = ["build_dataset_manifest", "load_radar_segment", "estimate_ppg_hr", "generate_ppg_labels", "WindowBounds", "crop_radar_window", "local_window_bounds"]
