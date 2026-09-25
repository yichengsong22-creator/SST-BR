"""Verify full package imports and lightweight core execution."""
from __future__ import annotations

import numpy as np

from sstbr.demo.synthetic import run_demo


def main() -> None:
    result = run_demo()
    if result["representation_shape"] != (6, 331) or result["context_shape"] != (1987,) or result["evaluation_samples"] != 3:
        raise RuntimeError("Unexpected demo output shape or sample count.")
    values = [result["baseline"], result["residual"], result["final"], *result["metrics"].values()]
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise RuntimeError("Demo output contains non-finite values.")
    print("Installation verified successfully.")


if __name__ == "__main__":
    main()
