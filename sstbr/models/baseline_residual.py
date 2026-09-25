"""Four-fold train-only fitting and direct-PPG 10-s evaluation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..data.io import read_fold
from ..evaluation.metrics import regression_metrics
from ..temporal.fusion import build_temporal_context
from ..temporal.fusion import fuse_one_subject


def baseline_model(settings: dict, criterion: str) -> Pipeline:
    """Build the SST-BR Extra-Trees subject-baseline pipeline."""
    params = dict(settings)
    params["criterion"] = criterion
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(**params)),
    ])


def residual_model(settings: dict, criterion: str) -> Pipeline:
    """Build the SST-BR Random-Forest window-residual pipeline."""
    params = dict(settings)
    params["criterion"] = criterion
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(**params)),
    ])


def residual_limits(training_residuals: np.ndarray) -> tuple[float, float]:
    """Apply the training-only percentile and hard-bound residual rule."""
    return (
        max(float(np.quantile(training_residuals, 0.01)), -18.0),
        min(float(np.quantile(training_residuals, 0.99)), 18.0),
    )


@dataclass
class BaselineResidualRegressor:
    """Reusable implementation of the SST-BR baseline–residual data flow."""

    baseline_settings: dict
    residual_settings: dict
    baseline_criterion: str
    residual_criterion: str

    def fit(self, features: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray) -> "BaselineResidualRegressor":
        """Fit baseline on subject contexts and residual on local windows."""
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float)
        subject_ids = np.asarray(subject_ids).astype(str)
        if features.ndim != 2 or labels.shape != (len(features),) or subject_ids.shape != (len(features),):
            raise ValueError("Expected features (rows, columns), labels (rows,), and subject_ids (rows,).")
        self.subject_order_ = np.unique(subject_ids)
        contexts = np.vstack([fuse_one_subject(features[subject_ids == subject]) for subject in self.subject_order_])
        medians = np.asarray([np.median(labels[subject_ids == subject]) for subject in self.subject_order_])
        self.baseline_ = baseline_model(self.baseline_settings, self.baseline_criterion).fit(contexts, medians)
        baseline_by_subject = dict(zip(self.subject_order_, self.baseline_.predict(contexts)))
        residual_targets = labels - np.asarray([baseline_by_subject[subject] for subject in subject_ids])
        self.residual_limits_ = residual_limits(residual_targets)
        self.residual_ = residual_model(self.residual_settings, self.residual_criterion).fit(features, residual_targets)
        return self

    def predict_baseline(self, temporal_context: np.ndarray) -> float:
        """Predict subject baseline HR from one fused temporal context."""
        return float(np.clip(self.baseline_.predict(np.asarray(temporal_context, float)[None, :])[0], 45.0, 130.0))

    def predict_residual(self, window_representation: np.ndarray) -> float:
        """Predict and training-bound one local residual."""
        value = self.residual_.predict(np.asarray(window_representation, float)[None, :])[0]
        return float(np.clip(value, *self.residual_limits_))

    @staticmethod
    def combine_hr(baseline: float, residual: float) -> float:
        """Combine baseline and residual using the SST-BR final HR bounds."""
        return float(np.clip(baseline + residual, 45.0, 130.0))


def _fold_subjects(metadata: pd.DataFrame, fold_path: Path) -> dict[str, set[str]]:
    raw = read_fold(fold_path)
    result = {}
    for split, segments in raw.items():
        subset = metadata.loc[metadata.source_segment_id.isin(set(map(str, segments)))]
        if subset.source_segment_id.nunique() != len(segments):
            raise ValueError(f"Fold segment mapping failed: {fold_path}/{split}")
        result[split] = set(subset.subject_id.astype(str))
    if tuple(len(result[x]) for x in ("train", "validation", "test")) != (52, 10, 20):
        raise ValueError(f"Fold subject counts: { {k: len(v) for k,v in result.items()} }")
    if any(result[a] & result[b] for a in result for b in result if a < b):
        raise ValueError("Fold subject leakage")
    return result


def train_and_predict(config: dict, features: np.ndarray, metadata: pd.DataFrame, labels5: pd.DataFrame, labels10: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Fit four fixed folds and return local plus 10-s evaluation predictions."""
    root = config["_project_root"]
    keys = ["subject_id", "source_segment_id", "local_window_id"]
    labels = labels5[keys + ["hr_bpm"]].copy()
    joined = metadata.merge(labels, on=keys, how="left", validate="one_to_one", indicator=True)
    if len(joined) != 1476 or not joined._merge.eq("both").all():
        raise ValueError("Feature/5-s PPG alignment failed")
    joined = joined.drop(columns="_merge")
    all_5s, fold_reports, decomposition = [], [], []
    for fold_index in range(1, 5):
        fold = f"Fold{fold_index}"
        members = _fold_subjects(metadata, root / config["paths"]["folds_dir"] / f"fold{fold_index}.yaml")
        split_by_subject = {subject: split for split, values in members.items() for subject in values}
        split = joined.subject_id.map(split_by_subject)
        context, context_meta, context_names = build_temporal_context(features, metadata, feature_names, split_by_subject)
        train_window = split.eq("train").to_numpy()
        train_subject = context_meta.split.eq("train").to_numpy()
        # Fit both branches using training subjects only; test labels are
        # attached later for evaluation.
        train_medians = joined.loc[train_window].groupby("subject_id", sort=True).hr_bpm.median()
        baseline_targets = context_meta.loc[train_subject, "subject_id"].map(train_medians).to_numpy(float)
        criteria = config["models"]["criteria"][f"fold{fold_index}"]
        baseline = baseline_model(config["models"]["baseline"], criteria["et_criterion"])
        started = time.perf_counter(); baseline.fit(context[train_subject], baseline_targets); baseline_seconds = time.perf_counter() - started
        baseline_predictions = np.clip(baseline.predict(context), 45.0, 130.0)
        baseline_by_subject = dict(zip(context_meta.subject_id, baseline_predictions))
        training_subject_baseline = joined.loc[train_window, "subject_id"].map(train_medians).to_numpy(float)
        residual_target = joined.loc[train_window, "hr_bpm"].to_numpy(float) - training_subject_baseline
        residual = residual_model(config["models"]["residual"], criteria["rf_criterion"])
        started = time.perf_counter(); residual.fit(features[train_window], residual_target); residual_seconds = time.perf_counter() - started
        lower, upper = residual_limits(residual_target)
        predicted_residual = np.clip(residual.predict(features), lower, upper)
        predicted_baseline = joined.subject_id.map(baseline_by_subject).to_numpy(float)
        prediction = np.clip(predicted_baseline + predicted_residual, 45.0, 130.0)
        test = split.eq("test").to_numpy()
        part = joined.loc[test, ["subject_id", "source_segment_id", "source_segment_ordinal", "local_window_id", "global_window_id", "hr_bpm"]].copy()
        part.insert(0, "fold", fold)
        part["baseline_prediction"] = predicted_baseline[test]
        part["residual_prediction"] = predicted_residual[test]
        part["pred_hr"] = prediction[test]
        part["gt_hr"] = part.pop("hr_bpm")
        part["error"] = part.pred_hr - part.gt_hr
        if len(part) != 360 or part.subject_id.nunique() != 20:
            raise ValueError(f"Unexpected {fold} test output size or subject count")
        model_dir = root / config["paths"]["models_dir"] / fold.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, model_dir / "subject_baseline_model.joblib")
        joblib.dump(residual, model_dir / "residual_model.joblib")
        all_5s.append(part)
        recon = np.clip(part.baseline_prediction + part.residual_prediction, 45.0, 130.0)
        decomposition.append({"fold": fold, "max_abs_reconstruction_error": float(np.max(np.abs(recon - part.pred_hr))), "residual_clip": [lower, upper]})
        fold_reports.append({
            "fold": fold, "train_subjects": 52, "validation_subjects": 10, "test_subjects": 20,
            "fit_scope": "train only", "validation_used_in_final_fit": False,
            "et_criterion": criteria["et_criterion"], "rf_criterion": criteria["rf_criterion"],
            "baseline_training_seconds": baseline_seconds, "residual_training_seconds": residual_seconds,
            "baseline_input_dim": int(baseline.named_steps["model"].n_features_in_), "residual_input_dim": int(residual.named_steps["model"].n_features_in_),
            "residual_clip": [lower, upper],
        })
    predictions5 = pd.concat(all_5s, ignore_index=True)
    if len(predictions5) != 1440 or predictions5.subject_id.nunique() != 80:
        raise ValueError("Unexpected pooled local-prediction count or subject count")
    predictions5["evaluation_window_id"] = predictions5.local_window_id // 2
    predictions10 = predictions5.groupby(["fold", "subject_id", "source_segment_id", "evaluation_window_id"], sort=False, as_index=False).agg(
        pred_hr=("pred_hr", "mean"), source_window_count=("local_window_id", "size"),
        source_window_ids=("local_window_id", lambda x: ",".join(map(str, sorted(x)))),
    )
    direct = labels10[["subject_id", "source_segment_id", "evaluation_window_id", "hr_bpm"]].rename(columns={"hr_bpm": "gt_hr"})
    predictions10 = predictions10.merge(direct, on=["subject_id", "source_segment_id", "evaluation_window_id"], validate="one_to_one")
    predictions10["error"] = predictions10.pred_hr - predictions10.gt_hr
    if len(predictions10) != 720 or not predictions10.groupby("fold").size().eq(180).all() or not predictions10.source_window_count.eq(2).all():
        raise ValueError("Unexpected 10-s prediction count or local-window pairing")
    metric_report = {"per_fold": {}, "pooled": regression_metrics(predictions10.gt_hr, predictions10.pred_hr)}
    for fold in ("Fold1", "Fold2", "Fold3", "Fold4"):
        part = predictions10.loc[predictions10.fold.eq(fold)]
        metric_report["per_fold"][fold] = regression_metrics(part.gt_hr, part.pred_hr)
    return predictions5, predictions10, metric_report, {"folds": fold_reports, "decomposition": decomposition, "context_dim": len(context_names)}
