"""Public feature-stage contracts and educational views."""

from .extractor import LocalRepresentationExtractor, WindowRepresentation
from .spatial import SpatialView
from .spectral import SpectralView

__all__ = ["LocalRepresentationExtractor", "WindowRepresentation", "SpatialView", "SpectralView"]
