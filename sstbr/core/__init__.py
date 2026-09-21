"""Protected exact-core interface and release-status error."""

from .interfaces import SSTBRCore
from .unavailable import CoreImplementationUnavailable

__all__ = ["SSTBRCore", "CoreImplementationUnavailable"]
