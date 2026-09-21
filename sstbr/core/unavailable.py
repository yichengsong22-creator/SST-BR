"""Explicit status error for the exact paper implementation."""


class CoreImplementationUnavailable(RuntimeError):
    """Raised only when exact paper-core execution is requested."""

    def __init__(self) -> None:
        super().__init__(
            "The exact SST-BR paper core is not included in this pre-publication release. "
            "Use DemoCore for the executable structural demonstration."
        )
