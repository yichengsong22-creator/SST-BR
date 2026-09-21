import numpy as np

from sstbr import DemoCore
from sstbr.demo.synthetic import generate_synthetic_recording


def test_demo_core_dataflow_is_deterministic() -> None:
    recording = generate_synthetic_recording()
    core = DemoCore()
    first = [core.extract_local_representation(window) for window in recording.radar_windows]
    second = [core.extract_local_representation(window) for window in recording.radar_windows]
    assert np.array_equal(first[0].values, second[0].values)
    context = core.fuse_temporal_context([item.values for item in first])
    baseline = core.predict_baseline(context)
    residual = core.predict_residual(first[-1])
    assert core.combine_hr(baseline, residual) == baseline + residual
    assert context.window_count == 6
