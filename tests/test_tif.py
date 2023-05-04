import pytest
import tifffile
import os
import numpy as np

from fibsem_tools.io.tif import access
from fibsem_tools import read


@pytest.mark.parametrize("memmap", (True, False))
def test_access(temp_dir, memmap):
    name = "test.tif"
    path = os.path.join(temp_dir, name)

    data = np.arange(27, dtype="uint8").reshape((3, 3, 3))
    tifffile.imwrite(path, data)

    observed = [access(path, mode="r", memmap=memmap), read(path, memmap=memmap)]

    if memmap:
        expected = tifffile.memmap(path)
    else:
        expected = tifffile.imread(path)

    for obs in observed:
        assert np.array_equal(obs, expected)
