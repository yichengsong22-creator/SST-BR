"""Run the deterministic partial-release structural demonstration."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.synthetic.generate_demo_data import generate_synthetic_recording
from sstbr import DemoCore
from sstbr.data.ppg_labels import estimate_ppg_hr
from sstbr.evaluation import aggregate_consecutive_predictions, attach_direct_references, regression_metrics


def run_demo() -> dict:
    """Exercise every public stage without requesting the exact paper core."""
    recording = generate_synthetic_recording()
    core = DemoCore()
    representations = [core.extract_local_representation(window) for window in recording.radar_windows]
    context = core.fuse_temporal_context([item.values for item in representations])
    baseline = core.predict_baseline(context)
    local_predictions = [
        core.combine_hr(baseline, core.predict_residual(representation))
        for representation in representations
    ]
    prediction_table = pd.DataFrame({
        "fold": ["synthetic"] * 6,
        "subject_id": ["synthetic"] * 6,
        "source_segment_id": ["synthetic-segment"] * 6,
        "local_window_id": range(6),
        "pred_hr": local_predictions,
    })
    direct = pd.DataFrame({
        "subject_id": ["synthetic"] * 3,
        "source_segment_id": ["synthetic-segment"] * 3,
        "evaluation_window_id": range(3),
        "hr_bpm": [
            estimate_ppg_hr(recording.ppg[index * 300:(index + 1) * 300])
            for index in range(3)
        ],
    })
    evaluation = attach_direct_references(aggregate_consecutive_predictions(prediction_table), direct)
    return {
        "baseline": baseline,
        "residual": core.predict_residual(representations[-1]),
        "final": local_predictions[-1],
        "local_windows": len(local_predictions),
        "evaluation_samples": len(evaluation),
        "metrics": regression_metrics(evaluation.gt_hr, evaluation.pred_hr),
    }


def main() -> None:
    result = run_demo()
    metrics = result["metrics"]
    print("SST-BR pre-publication executable demonstration")
    print("Synthetic demonstration only - not a paper-result reproduction")
    print(f"Local windows: {result['local_windows']}")
    print(f"10-s evaluation samples: {result['evaluation_samples']}")
    print(f"Baseline: {result['baseline']:.3f} bpm")
    print(f"Residual: {result['residual']:.3f} bpm")
    print(f"Final estimate: {result['final']:.3f} bpm")
    print(f"RMSE: {metrics['RMSE']:.3f} bpm | MAE: {metrics['MAE']:.3f} bpm | r: {metrics['r']:.3f}")
    print("Exact paper core included: NO")


if __name__ == "__main__":
    main()
