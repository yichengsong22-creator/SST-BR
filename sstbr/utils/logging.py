"""Consistent command-line logging."""
from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise logging for examples and command-line utilities."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
