"""Lightweight example wrapper around the installed SST-BR demo package."""

from sstbr.demo.synthetic import generate_synthetic_recording


def main() -> None:
    """Generate one in-memory recording and print its public shapes."""
    recording = generate_synthetic_recording()
    print("radar_windows:", recording.radar_windows.shape)
    print("range_grid:", recording.range_grid.shape)
    print("ppg:", recording.ppg.shape)


if __name__ == "__main__":
    main()
