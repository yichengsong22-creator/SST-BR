import numpy as np
import pytest

from sstbr import CoreImplementationUnavailable, SSTBRCore


def test_exact_core_interface_is_explicitly_protected() -> None:
    core = SSTBRCore()
    with pytest.raises(CoreImplementationUnavailable, match="pre-publication"):
        core.extract_local_representation(np.zeros((2, 8)))
