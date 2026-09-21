"""Public configuration and raw RHB-style data adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat


@dataclass(frozen=True)
class RawSegment:
    """Paths and neutral identifiers for one recorded source segment."""

    subject_id: str
    source_segment_id: str
    source_segment_ordinal: int
    radar_path: Path
    ppg_path: Path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration without embedding machine-specific paths."""
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "dataset" not in config:
        raise ValueError("Configuration must contain a 'dataset' mapping.")
    root = config["dataset"].get("rhb_root")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("dataset.rhb_root must be a non-empty path string.")
    config["_config_path"] = config_path
    config["dataset"]["rhb_root"] = Path(root).expanduser()
    return config


def _segment_key(name: str) -> tuple[int, int]:
    try:
        subject, ordinal = name.rsplit("_", 1)
        return int(subject), int(ordinal)
    except ValueError as error:
        raise ValueError(f"Expected source segment name '<subject>_<ordinal>', got {name!r}.") from error


def discover_raw_segments(root: str | Path) -> list[RawSegment]:
    """Discover source segments, requiring one radar MAT and one PPG NPY file each."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(root)
    segments: list[RawSegment] = []
    directories = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: _segment_key(path.name))
    for directory in directories:
        subject, ordinal = _segment_key(directory.name)
        radar_candidates = sorted(directory.glob("*complex_range_matrix*.mat")) or sorted(directory.glob("*.mat"))
        ppg_path = directory / "vital_dict.npy"
        if len(radar_candidates) != 1:
            raise ValueError(f"Expected exactly one radar MAT file in {directory}.")
        if not ppg_path.is_file():
            raise FileNotFoundError(ppg_path)
        segments.append(RawSegment(str(subject), directory.name, ordinal, radar_candidates[0], ppg_path))
    if not segments:
        raise ValueError(f"No source segments found below {root}.")
    keys = [(item.subject_id, item.source_segment_id) for item in segments]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source-segment key detected.")
    return segments


def _mat_field(value: Any, name: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, dict):
        return value.get(name)
    if isinstance(value, np.ndarray) and value.dtype.names and name in value.dtype.names:
        return np.squeeze(value)[name]
    return None


def load_radar_segment(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a complex radar matrix as ``(range_bins, slow_time_samples)``."""
    path = Path(path)
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    container = raw.get("save_data", raw.get("segment_data"))
    matrix = _mat_field(container, "complex_range_matrix") if container is not None else raw.get("complex_range_matrix")
    range_grid = _mat_field(container, "range_grid") if container is not None else raw.get("range_grid")
    if matrix is None or range_grid is None:
        raise KeyError(f"Missing complex_range_matrix or range_grid in {path}.")
    matrix = np.asarray(matrix)
    range_grid = np.asarray(range_grid, dtype=float).reshape(-1)
    if matrix.ndim != 2 or range_grid.size not in matrix.shape:
        raise ValueError(f"Invalid radar matrix/grid shapes: {matrix.shape}/{range_grid.shape}.")
    if matrix.shape[0] != range_grid.size:
        matrix = matrix.T
    if not (np.isfinite(matrix.real).all() and np.isfinite(matrix.imag).all() and np.isfinite(range_grid).all()):
        raise ValueError(f"Non-finite radar input in {path}.")
    return matrix, range_grid


def load_ppg_signal(path: str | Path, required_samples: int | None = None) -> np.ndarray:
    """Load a finite one-dimensional PPG signal, optionally enforcing length."""
    path = Path(path)
    signal = np.asarray(np.load(path, allow_pickle=True))
    if signal.ndim != 1:
        raise ValueError(f"PPG must be one-dimensional: {path} has shape {signal.shape}.")
    signal = np.asarray(signal, dtype=np.float64)
    if required_samples is not None and len(signal) < required_samples:
        raise ValueError(f"PPG is shorter than {required_samples} samples: {path}.")
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite PPG input in {path}.")
    return signal
