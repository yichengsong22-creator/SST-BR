"""Public data, windowing, and PPG-label utilities."""

from .io import RawSegment, discover_raw_segments, load_ppg_signal, load_radar_segment
from .ppg_labels import estimate_ppg_hr, generate_ppg_label_tables
from .windowing import WindowSpec, crop_available_window, make_window_specs

__all__ = ["RawSegment", "discover_raw_segments", "load_ppg_signal", "load_radar_segment", "estimate_ppg_hr", "generate_ppg_label_tables", "WindowSpec", "crop_available_window", "make_window_specs"]
