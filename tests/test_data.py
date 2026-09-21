from pathlib import Path

import numpy as np
import pytest

from sstbr.data import RawSegment, estimate_ppg_hr, generate_ppg_label_tables, make_window_specs


def test_windowing_and_ppg_labels(tmp_path: Path) -> None:
    assert len(make_window_specs(30.0, 5.0, 120.0)) == 6
    time = np.arange(900) / 30.0
    ppg = np.sin(2 * np.pi * 1.3 * time)
    assert estimate_ppg_hr(ppg[:300]) == pytest.approx(78.0, abs=0.2)
    ppg_path = tmp_path / "vital_dict.npy"
    np.save(ppg_path, ppg)
    segment = RawSegment("demo", "demo_1", 1, tmp_path / "unused.mat", ppg_path)
    local, direct = generate_ppg_label_tables([segment])
    assert (len(local), len(direct)) == (6, 3)
