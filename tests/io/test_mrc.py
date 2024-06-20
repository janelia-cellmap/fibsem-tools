import os

import mrcfile
import numpy as np
import pytest
from xarray.testing import assert_equal

from fibsem_tools.coordinate import stt_array
from fibsem_tools.io.core import read_xarray
from fibsem_tools.io.mrc import (
    MrcArrayWrapper,
    access,
    recarray_to_dict,
    to_dask,
    to_xarray,
)


def test_access_mrc(tmpdir):
    name = "test.mrc"
    mrc_path = os.path.join(str(tmpdir), name)

    data = np.arange(27, dtype="uint8").reshape((3, 3, 3))
    original = mrcfile.new(mrc_path, data=data, overwrite=True)
    original.flush()
    accessed = access(mrc_path, mode="r")
    assert np.array_equal(accessed.mrc.data, original.data)
    assert np.array_equal(to_dask(accessed).compute(), original.data)
    assert np.array_equal(
        to_dask(accessed, chunks=(2, -1, -1)).compute(), original.data
    )

    with pytest.raises(ValueError):
        to_dask(accessed, chunks=(1, 1, 1))

    del original
    del accessed


@pytest.mark.parametrize("attrs", [None, {"foo": 10}])
def test_read_xarray(tmpdir, attrs):
    name = "test.mrc"
    mrc_path = os.path.join(tmpdir, name)
    scales = [1.0, 2.0, 3.0]
    data = np.arange(4 * 5 * 6, dtype="uint8").reshape((4, 5, 6))
    original = mrcfile.new(mrc_path, data=data, overwrite=True)
    original.voxel_size = [x * 10 for x in reversed(scales)]
    original.flush()

    expected = stt_array(
        data,
        dims=("z", "y", "x"),
        translates=(0, 0, 0),
        scales=scales,
        units=("nm", "nm", "nm"),
    )

    observed = [
        to_xarray(MrcArrayWrapper(original), attrs=attrs),
        read_xarray(mrc_path, attrs=attrs),
    ]
    for obs in observed:
        assert_equal(obs, expected)
        if attrs is None:
            assert dict(obs.attrs) == recarray_to_dict(original.header)
