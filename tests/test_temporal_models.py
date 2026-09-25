import numpy as np

from sstbr.models import BaselineResidualRegressor
from sstbr.temporal import fuse_one_subject


def test_temporal_fusion_shape() -> None:
    features = np.arange(24, dtype=float).reshape(4, 6)
    context = fuse_one_subject(features)
    assert context.shape == (37,)
    assert context[0] == 4


def test_baseline_residual_fit_and_combination() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(48, 5))
    subjects = np.repeat([f"S{i}" for i in range(8)], 6)
    labels = 70 + np.repeat(np.arange(8), 6) + 2 * features[:, 0]
    settings = {"n_estimators": 12, "random_state": 3, "n_jobs": 1, "min_samples_leaf": 1}
    model = BaselineResidualRegressor(settings, settings, "squared_error", "squared_error").fit(features, labels, subjects)
    context = fuse_one_subject(features[subjects == "S0"])
    baseline = model.predict_baseline(context)
    residual = model.predict_residual(features[0])
    assert np.isfinite(model.combine_hr(baseline, residual))
