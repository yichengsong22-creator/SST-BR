"""Run a lightweight synthetic example through the full public core."""
from __future__ import annotations

from sstbr.demo.synthetic import run_demo


def main() -> None:
    result = run_demo()
    metrics = result["metrics"]
    print("SST-BR full-release synthetic demonstration")
    print("Synthetic demonstration only - not an RHB result")
    print(f"Representation batch: {result['representation_shape']}")
    print(f"Temporal context: {result['context_shape']}")
    print(f"Local windows: {result['local_windows']}")
    print(f"10-s evaluation samples: {result['evaluation_samples']}")
    print(f"Baseline: {result['baseline']:.3f} bpm")
    print(f"Residual: {result['residual']:.3f} bpm")
    print(f"Final estimate: {result['final']:.3f} bpm")
    print(f"RMSE: {metrics['RMSE']:.3f} bpm | MAE: {metrics['MAE']:.3f} bpm | r: {metrics['r']:.3f}")
    print("Full SST-BR core included: YES")


if __name__ == "__main__":
    main()
