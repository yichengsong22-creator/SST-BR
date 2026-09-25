import numpy as np
import pytest

from sstbr.data.ppg_labels import estimate_ppg_hr
from sstbr.data.windowing import crop_radar_window, local_window_bounds


def test_windowing_and_short_final_crop() -> None:
    bounds = local_window_bounds(5)
    assert (bounds.radar_start, bounds.radar_end_nominal) == (3000, 3600)
    matrix = np.zeros((4, 3598), dtype=complex)
    window, start, end = crop_radar_window(matrix, 5)
    assert window.shape == (4, 598)
    assert (start, end) == (3000, 3598)


def test_ppg_hr_generator() -> None:
    time = np.arange(300) / 30.0
    ppg = np.sin(2 * np.pi * 1.2 * time)
    assert estimate_ppg_hr(ppg) == pytest.approx(72.0, abs=0.2)
