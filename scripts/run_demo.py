"""Run the deterministic partial-release structural demonstration."""
from __future__ import annotations

from sstbr.demo.synthetic import run_structural_demo


def main() -> None:
    result = run_structural_demo()
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
