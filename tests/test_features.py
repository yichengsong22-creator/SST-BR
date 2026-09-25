import numpy as np

from sstbr import SSTBRCore


def _deterministic_radar_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Return one small in-memory radar window for a feature invariant test."""
    rng = np.random.default_rng(2026)
    radar_hz = 120.0
    time = np.arange(int(5.0 * radar_hz)) / radar_hz
    range_grid = np.linspace(0.25, 2.25, 12)
    envelope = np.exp(-0.5 * ((np.arange(12) - 4) / 1.3) ** 2)[:, None]
    phase = 0.55 * np.sin(2.0 * np.pi * (78.0 / 60.0) * time)
    signal = envelope * np.exp(1j * phase)[None, :]
    noise = 0.08 * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    return signal + noise, range_grid


def test_complete_spatial_spectral_representation() -> None:
    radar_window, range_grid = _deterministic_radar_fixture()
    core = SSTBRCore(radar_fs=120.0)
    representation = core.extract_local_representation(radar_window, range_grid)
    assert representation.shape == (331,)
    assert np.isfinite(representation).sum() > 300
