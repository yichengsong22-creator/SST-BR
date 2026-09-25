"""Complete raw-RHB-to-10-second SST-BR reproduction pipeline."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data.io import build_dataset_manifest
from .data.ppg_labels import generate_ppg_labels
from .evaluation.metrics import regression_metrics
from .features.extractor import extract_all_features, load_feature_contract
from .models.baseline_residual import train_and_predict
from .utils.config import config_hash, load_config
from .utils.files import read_json, stable_hash, write_json


def _prepare_labels(config: dict, manifest: pd.DataFrame, reuse: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = config["_project_root"]
    folder = root / config["paths"]["ppg_cache_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    local_path = folder / "ppg_labels_local.csv"
    direct_path = folder / "ppg_labels_10s_direct.csv"
    marker_path = folder / "cache_metadata.json"
    expected = {
        "pipeline_version": __version__,
        "dataset_manifest_hash": stable_hash(manifest.to_dict("records")),
        "config_hash": config_hash(config),
    }
    valid_cache = (
        reuse
        and local_path.is_file()
        and direct_path.is_file()
        and marker_path.is_file()
        and all(read_json(marker_path).get(key) == value for key, value in expected.items())
    )
    if valid_cache:
        local = pd.read_csv(local_path)
        direct = pd.read_csv(direct_path)
    else:
        local, direct = generate_ppg_labels(manifest)
        local.to_csv(local_path, index=False, float_format="%.17g")
        direct.to_csv(direct_path, index=False, float_format="%.17g")
        write_json(marker_path, {**expected, "local_rows": len(local), "direct_rows": len(direct)})
        local = pd.read_csv(local_path)
        direct = pd.read_csv(direct_path)
    for table in (local, direct):
        table["subject_id"] = table.subject_id.astype(str)
        table["source_segment_id"] = table.source_segment_id.astype(str)
    return local, direct


def _prepare_features(
    config: dict,
    manifest: pd.DataFrame,
    reuse: bool,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    root = config["_project_root"]
    folder = root / config["paths"]["radar_cache_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    array_path = folder / "window_features.npz"
    metadata_path = folder / "window_metadata.csv"
    marker_path = folder / "cache_metadata.json"
    feature_manifest, _ = load_feature_contract(root)
    names = [item["feature_name"] for item in feature_manifest]
    expected = {
        "pipeline_version": __version__,
        "feature_names_hash": stable_hash(names),
        "config_hash": config_hash(config),
        "dataset_manifest_hash": stable_hash(manifest.to_dict("records")),
    }
    valid_cache = (
        reuse
        and array_path.is_file()
        and metadata_path.is_file()
        and marker_path.is_file()
        and all(read_json(marker_path).get(key) == value for key, value in expected.items())
    )
    if valid_cache:
        features = np.load(array_path)["window_features"]
        metadata = pd.read_csv(metadata_path, dtype={"subject_id": str, "source_segment_id": str})
    else:
        features, metadata, names, extraction_audit = extract_all_features(
            root, config["dataset"]["rhb_root"], manifest
        )
        # Apply the established CSV-precision normalization before training.
        for _ in range(2):
            buffer = io.StringIO()
            pd.DataFrame(features, columns=names).to_csv(buffer, index=False, float_format="%.17g")
            buffer.seek(0)
            features = pd.read_csv(buffer)[names].to_numpy(dtype=np.float64)
        np.savez_compressed(array_path, window_features=features)
        metadata.to_csv(metadata_path, index=False)
        write_json(marker_path, {**expected, "shape": list(features.shape)})
        write_json(root / "outputs" / "radar_extraction_audit.json", extraction_audit)
    if features.shape != (1476, 331) or len(metadata) != 1476:
        raise ValueError(f"Unexpected full feature shape: {features.shape}/{metadata.shape}")
    return features, metadata, names


def run_full_pipeline(config_path: str | Path, reuse_cache: bool = False) -> dict:
    """Run the complete SST-BR pipeline and return its metric report."""
    config = load_config(config_path)
    root = config["_project_root"]
    manifest = build_dataset_manifest(config["dataset"]["rhb_root"])
    output_dir = root / config["paths"]["outputs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)

    labels_local, labels_direct = _prepare_labels(config, manifest, reuse_cache)
    features, metadata, feature_names = _prepare_features(config, manifest, reuse_cache)
    predictions_local, predictions_10s, metrics, training = train_and_predict(
        config, features, metadata, labels_local, labels_direct, feature_names
    )
    predictions_local.to_csv(output_dir / "predictions_local.csv", index=False, float_format="%.17g")
    predictions_10s.to_csv(output_dir / "predictions_10s.csv", index=False, float_format="%.17g")
    write_json(output_dir / "metrics_10s.json", metrics)
    write_json(output_dir / "training_report.json", training)
    write_json(output_dir / "local_metrics.json", regression_metrics(predictions_local.gt_hr, predictions_local.pred_hr))

    return metrics
