import numpy as np

from sstbr import SSTBRCore
from sstbr.demo.synthetic import generate_synthetic_recording


def test_complete_spatial_spectral_representation() -> None:
    recording = generate_synthetic_recording()
    core = SSTBRCore(radar_fs=recording.radar_hz)
    representation = core.extract_local_representation(recording.radar_windows[0], recording.range_grid)
    assert representation.shape == (331,)
    assert np.isfinite(representation).sum() > 300
