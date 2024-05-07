from __future__ import annotations

import pytest
import tifffile
import os
import numpy as np

from fibsem_tools.io.tif import access
from fibsem_tools import read


@pytest.mark.parametrize("file_name", ("test.tif", "test.tiff"))
@pytest.mark.parametrize("memmap", (True, False))
def test_access(tmpdir, file_name: str, memmap: bool) -> None:
    path = os.path.join(str(tmpdir), file_name)

    data = np.arange(27, dtype="uint8").reshape((3, 3, 3))
    tifffile.imwrite(path, data)

    observed = [access(path, mode="r", memmap=memmap), read(path, memmap=memmap)]

    if memmap:
        expected = tifffile.memmap(path)
    else:
        expected = tifffile.imread(path)

    for obs in observed:
        assert np.array_equal(obs, expected)
