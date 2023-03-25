import dask.array as da
from fibsem_tools.io.dask import (
    copy_array,
    ensure_minimum_chunksize,
    autoscale_chunk_shape,
)
import pytest
import zarr
import numpy as np
import dask


def test_ensure_minimum_chunksize():
    data = da.zeros((10,), chunks=(2,))
    assert ensure_minimum_chunksize(data, (4,)).chunksize == (4,)

    data = da.zeros((10,), chunks=(6,))
    assert ensure_minimum_chunksize(data, (4,)).chunksize == (6,)

    data = da.zeros((10, 10), chunks=(2, 1))
    assert ensure_minimum_chunksize(data, (4, 4)).chunksize == (4, 4)

    data = da.zeros((10, 10, 10), chunks=(2, 1, 10))
    assert ensure_minimum_chunksize(data, (4, 4, 4)).chunksize == (4, 4, 10)


def test_autoscale_chunk_shape():
    chunk_shape = (1,)
    array_shape = (1000,)
    size_limit = "1KB"
    dtype = "uint8"

    assert autoscale_chunk_shape(chunk_shape, array_shape, size_limit, dtype) == (999,)

    chunk_shape = (1, 1)
    array_shape = (1000, 100)

    assert autoscale_chunk_shape(chunk_shape, array_shape, size_limit, dtype) == (
        999,
        1,
    )

    chunk_shape = (1, 1)
    array_shape = (1000, 1001)

    assert autoscale_chunk_shape(chunk_shape, array_shape, size_limit, dtype) == (
        1,
        999,
    )

    chunk_shape = (64, 64, 64)
    size_limit = "8MB"
    array_shape = (1000, 1000, 1000)

    assert autoscale_chunk_shape(chunk_shape, array_shape, size_limit, dtype) == (
        64,
        64,
        1024,
    )


@pytest.mark.parametrize("shape", ((1000,), (100, 100)))
def test_array_copy_from_array(temp_zarr, shape):
    g = zarr.group(zarr.NestedDirectoryStore(temp_zarr))
    arr_1 = g.create_dataset(name="a", data=np.random.randint(0, 255, shape))
    arr_2 = g.create_dataset(name="b", data=np.zeros(arr_1.shape, dtype=arr_1.dtype))

    copy_op = copy_array(arr_1, arr_2)
    dask.compute(copy_op)
    assert np.array_equal(arr_2, arr_1)


@pytest.mark.parametrize("shape", ((1000,), (100, 100)))
def test_array_copy_from_path(temp_zarr, shape):
    g = zarr.group(zarr.NestedDirectoryStore(temp_zarr))
    arr_1 = g.create_dataset(name="a", data=np.random.randint(0, 255, shape))
    arr_2 = g.create_dataset(name="b", data=np.zeros(arr_1.shape, dtype=arr_1.dtype))

    copy_op = copy_array(arr_1, arr_2)
    dask.compute(copy_op)
    assert np.array_equal(arr_2, arr_1)
