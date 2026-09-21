"""Public local-representation data structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class WindowRepresentation:
    """Validated one-dimensional representation of one local radar window."""

    values: np.ndarray
    source: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("WindowRepresentation values must be finite and one-dimensional.")
        object.__setattr__(self, "values", values)


class LocalRepresentationExtractor(Protocol):
    """Shape-level interface shared by demo and future full implementations."""

    def extract_local_representation(self, radar_window: np.ndarray) -> WindowRepresentation:
        """Map one `(range_bins, samples)` window to a local representation."""
