"""Public baseline–residual data-flow objects."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineResidualEstimate:
    """Transparent decomposition of one educational HR estimate."""

    baseline: float
    residual: float

    @property
    def heart_rate(self) -> float:
        """Combine the two branches without a hidden post-processing rule."""
        return float(self.baseline + self.residual)


def combine_hr(baseline: float, residual: float) -> float:
    """Expose the method's high-level additive combination."""
    return BaselineResidualEstimate(float(baseline), float(residual)).heart_rate
