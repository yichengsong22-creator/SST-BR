"""Strict raw RHB inventory, radar loading and fixed fold reading."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat

from .windowing import TOTAL_SOURCE_SEGMENTS, TOTAL_SUBJECTS


def segment_sort_key(value: str) -> tuple[int, int]:
    subject, ordinal = str(value).rsplit("_", 1)
    return int(subject), int(ordinal)


def build_dataset_manifest(rhb_root: Path) -> pd.DataFrame:
    """Validate the 82-subject/246-segment disk inventory without reading labels."""
    rows = []
    for directory in sorted((p for p in rhb_root.iterdir() if p.is_dir()), key=lambda p: segment_sort_key(p.name)):
        subject, ordinal = segment_sort_key(directory.name)
        if ordinal not in (1, 2, 3):
            raise ValueError(f"Unexpected source segment name: {directory.name}")
        radar = sorted(directory.glob("*complex_range_matrix*.mat")) or sorted(directory.glob("*.mat"))
        vital = directory / "vital_dict.npy"
        if len(radar) != 1 or not vital.is_file():
            raise FileNotFoundError(f"Expected one radar MAT and vital_dict.npy in {directory}")
        rows.append({
            "subject_id": str(subject), "source_segment_id": directory.name,
            "source_segment_ordinal": ordinal, "radar_file": str(radar[0]), "ppg_file": str(vital),
            "radar_size_bytes": radar[0].stat().st_size, "ppg_size_bytes": vital.stat().st_size,
        })
    frame = pd.DataFrame(rows)
    counts = (frame.subject_id.nunique(), len(frame))
    if counts != (TOTAL_SUBJECTS, TOTAL_SOURCE_SEGMENTS):
        raise ValueError(f"Raw dataset cardinality mismatch: {counts}")
    if not frame.groupby("subject_id").size().eq(3).all() or frame.duplicated("source_segment_id").any():
        raise ValueError("Each subject must have exactly three uniquely keyed source segments")
    return frame


def _field(obj, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return np.squeeze(obj)[name]
    return None


def load_radar_segment(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Read save_data.complex_range_matrix and normalize to (range_bin, slow_time)."""
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    obj = raw.get("save_data", raw.get("segment_data"))
    matrix = _field(obj, "complex_range_matrix") if obj is not None else raw.get("complex_range_matrix")
    grid = _field(obj, "range_grid") if obj is not None else raw.get("range_grid")
    if matrix is None or grid is None:
        raise KeyError(f"Missing complex_range_matrix/range_grid: {path}")
    matrix = np.asarray(matrix)
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if matrix.ndim != 2 or grid.size not in matrix.shape:
        raise ValueError(f"Invalid radar matrix/grid: {path} {matrix.shape}/{grid.shape}")
    if matrix.shape[0] != grid.size:
        matrix = matrix.T
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all() or not np.isfinite(grid).all():
        raise ValueError(f"Non-finite radar input: {path}")
    return matrix, grid, 120.0


def read_fold(path: Path) -> dict[str, list[str]]:
    """Read a public YAML subject-disjoint fold definition."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Fold definition must be a mapping: {path}")
    result: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        values = raw.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Missing or empty {split!r} split in {path}")
        result[split] = [str(item) for item in values]
    return result
