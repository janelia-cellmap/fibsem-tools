import os
import mrcfile
import numpy as np
import pytest
from fibsem_tools.io.xr import stt_from_array
from fibsem_tools.io.mrc import access, recarray_to_dict, to_dask, to_xarray
from xarray.testing import assert_equal


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


@pytest.mark.parametrize("attrs", (None, {"foo": 10}))
def test_read_xarray(temp_dir, attrs):
    name = "test.mrc"
    mrc_path = os.path.join(temp_dir, name)
    scales = [1.0, 2.0, 3.0]
    data = np.arange(4 * 5 * 6, dtype="uint8").reshape((4, 5, 6))
    original = mrcfile.new(mrc_path, data=data, overwrite=True)
    original.voxel_size = [x * 10 for x in reversed(scales)]
    original.flush()

    expected = stt_from_array(
        data,
        dims=("z", "y", "x"),
        translates=(0, 0, 0),
        scales=scales,
        units=("nm", "nm", "nm"),
    )

    observed = to_xarray(original, attrs=attrs)
    assert_equal(observed, expected)
    if attrs is None:
        assert dict(observed.attrs) == recarray_to_dict(original.header)
