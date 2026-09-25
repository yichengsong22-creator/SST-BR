"""Raw radar to the final ordered 331-D local representation."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from ..data.io import load_radar_segment
from .preprocessing import SignalRuntime, WindowComputationCache
from ..utils.files import read_json, stable_hash
from ..data.windowing import crop_radar_window, TOTAL_LOCAL_WINDOWS
from .spatial import SpatialExtractor
from .spectral import SpectralExtractor

EXPECTED_BLOCKS = {"original": 214, "energy": 48, "tracking": 36, "filterbank": 33}


def load_feature_contract(project_root: Path) -> tuple[list[dict], dict[str, list[str]]]:
    """Load extraction manifests and validate the final ordered 331 feature names."""
    folder = project_root / "configs" / "features"
    final_manifest = read_json(folder / "feature_manifest_331.json")
    final_names = [item["feature_name"] for item in final_manifest]
    counts = pd.Series([item["module"] for item in final_manifest]).value_counts().to_dict()
    if len(final_names) != 331 or len(set(final_names)) != 331 or counts != EXPECTED_BLOCKS:
        raise ValueError(f"331-D feature schema mismatch: {len(final_names)}/{counts}")
    extraction = {
        "original": read_json(folder / "original_291.json"),
        "energy": read_json(folder / "energy_54.json"),
        "tracking": read_json(folder / "tracking_39.json"),
        "filterbank": read_json(folder / "filterbank_40.json"),
    }
    return final_manifest, extraction


def extract_all_features(project_root: Path, dataset_root: Path, manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str], dict]:
    """Extract features from raw MAT files; returns (1476,331), metadata and audit."""
    final_manifest, names = load_feature_contract(project_root)
    feature_names = [item["feature_name"] for item in final_manifest]
    cache = WindowComputationCache()
    runtime = SignalRuntime(cache)
    spatial = SpatialExtractor(runtime, dataset_root, names["original"], names["energy"], names["tracking"])
    spectral = SpectralExtractor(runtime, names["filterbank"])
    rows, meta_rows, extraction_audit = [], [], []
    for subject_id, subject_sources in manifest.groupby("subject_id", sort=False):
        cache.begin_session()
        logging.info("Radar features: subject %s/82", subject_id)
        loaded = {row.source_segment_id: load_radar_segment(Path(row.radar_file)) for row in subject_sources.itertuples(index=False)}
        subject_spatial: list[dict] = []
        subject_metadata: list[dict] = []
        for source in subject_sources.sort_values("source_segment_ordinal").itertuples(index=False):
            matrix, grid, fs = loaded[source.source_segment_id]
            for local_window_id in range(6):
                window, start, end = crop_radar_window(matrix, local_window_id)
                ids = f"{source.source_segment_id}/{local_window_id}"
                spatial_values, audit = spatial.extract(ids, source.source_segment_id, local_window_id, window, grid, matrix.shape[1], fs, (matrix, grid))
                working = {"ids": ids, "subject_id": str(subject_id), "source_segment_id": source.source_segment_id, "local_window_id": local_window_id, **spatial_values}
                subject_spatial.append(working)
                global_id = (int(source.source_segment_ordinal) - 1) * 6 + local_window_id
                subject_metadata.append({
                    "ids": ids, "subject_id": str(subject_id), "source_segment_id": source.source_segment_id,
                    "source_segment_ordinal": int(source.source_segment_ordinal), "local_window_id": local_window_id,
                    "global_window_id": global_id, "subject_absolute_start_sec": global_id * 5.0,
                    "radar_start": start, "radar_end": end, "actual_length": end - start,
                })
                extraction_audit.append({"ids": ids, **audit})
        # Complete subject-level spatial extraction before reusing the shared
        # cache for spectral feature construction.
        for working, row_metadata in zip(subject_spatial, subject_metadata):
            matrix, _grid, fs = loaded[working["source_segment_id"]]
            spectral_values = spectral.extract(working, matrix, fs)
            all_values = {**working, **spectral_values}
            rows.append([all_values[name] for name in feature_names])
            meta_rows.append(row_metadata)
    matrix = np.asarray(rows, dtype=np.float64)
    metadata = pd.DataFrame(meta_rows)
    if matrix.shape != (TOTAL_LOCAL_WINDOWS, 331) or len(metadata) != TOTAL_LOCAL_WINDOWS:
        raise ValueError(f"Unexpected feature or metadata shape: {matrix.shape}/{metadata.shape}")
    if np.isinf(matrix).any() or metadata.duplicated(["subject_id", "source_segment_id", "local_window_id"]).any():
        raise ValueError("Infinite feature or duplicate feature key")
    if not metadata.groupby("subject_id").size().eq(18).all():
        raise ValueError("Each subject must have 18 feature rows")
    audit = {
        "shape": list(matrix.shape), "nan_count": int(np.isnan(matrix).sum()),
        "feature_names_hash": stable_hash(feature_names),
        "radar_window_lengths": sorted(metadata.actual_length.unique().astype(int).tolist()),
        "rows": extraction_audit,
    }
    return matrix, metadata, feature_names, audit
