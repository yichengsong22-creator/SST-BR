"""Generate the deterministic in-memory recording used by SST-BR examples."""
from __future__ import annotations

from sstbr.demo.synthetic import generate_synthetic_recording


def main() -> None:
    recording = generate_synthetic_recording()
    print("radar_windows:", recording.radar_windows.shape)
    print("range_grid:", recording.range_grid.shape)
    print("ppg:", recording.ppg.shape)


if __name__ == "__main__":
    main()
