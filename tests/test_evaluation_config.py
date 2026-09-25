from pathlib import Path

import pandas as pd
import pytest

from sstbr.evaluation import aggregate_consecutive_predictions, regression_metrics
from sstbr.utils.config import load_config


def test_aggregation_and_metrics() -> None:
    frame = pd.DataFrame({
        "subject_id": ["A"] * 4,
        "source_segment_id": ["A_1"] * 4,
        "local_window_id": [0, 1, 2, 3],
        "pred_hr": [60.0, 64.0, 80.0, 90.0],
    })
    aggregated = aggregate_consecutive_predictions(frame)
    assert aggregated.pred_hr.tolist() == pytest.approx([62.0, 85.0])
    metrics = regression_metrics([60.0, 80.0], aggregated.pred_hr)
    assert metrics["RMSE"] == pytest.approx((14.5) ** 0.5)


def test_config_loader_accepts_explicit_dataset_root(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(f'dataset:\n  rhb_root: "{tmp_path.as_posix()}"\n', encoding="utf-8")
    loaded = load_config(config)
    assert loaded["dataset"]["rhb_root"] == tmp_path.resolve()
