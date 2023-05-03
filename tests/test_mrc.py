import os
import mrcfile
import numpy as np
import pytest
from fibsem_tools.io.mrc import access, to_dask


def test_access_mrc(temp_dir):
    name = "test.mrc"
    mrc_path = os.path.join(temp_dir, name)

    data = np.arange(27, dtype="uint8").reshape((3, 3, 3))
    original = mrcfile.new(mrc_path, data=data, overwrite=True)
    original.flush()
    accessed = access(mrc_path, mode="r")
    assert np.array_equal(accessed.data, original.data)
    assert np.array_equal(to_dask(accessed).compute(), original.data)
    assert np.array_equal(
        to_dask(accessed, chunks=(2, -1, -1)).compute(), original.data
    )

    with pytest.raises(ValueError):
        to_dask(accessed, chunks=(1, 1, 1))

    del original
    del accessed


# todo: add dataarray tests
