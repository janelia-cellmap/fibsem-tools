from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from fibsem_tools import read
from fibsem_tools.io.tif import access


@pytest.mark.parametrize("file_name", ["test.tif", "test.tiff"])
@pytest.mark.parametrize("memmap", [True, False])
def test_access(tmpdir, file_name: str, memmap: bool) -> None:
    path = os.path.join(str(tmpdir), file_name)

    data = np.arange(27, dtype="uint8").reshape((3, 3, 3))
    tifffile.imwrite(path, data)

    observed = [access(path, mode="r", memmap=memmap), read(path, memmap=memmap)]

    expected = tifffile.memmap(path) if memmap else tifffile.imread(path)

    for obs in observed:
        assert np.array_equal(obs, expected)
