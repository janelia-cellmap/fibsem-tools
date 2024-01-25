from typing import Tuple
import dask.array as da
from fibsem_tools.io.dask import (
    copy_array,
    ensure_minimum_chunksize,
    autoscale_chunk_shape,
    interval_remainder,
    resolve_slice,
    resolve_slices,
    setitem,
    write_blocks_delayed,
)
import pytest
import zarr
import numpy as np
import dask
from pydantic_zarr import ArraySpec, GroupSpec
from numpy.testing import assert_array_equal
from dask.array.core import slices_from_chunks


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


@pytest.mark.parametrize("keep_attrs", (True, False))
@pytest.mark.parametrize("shape", ((10,), (10, 10)))
def test_array_copy_from_array(shape, keep_attrs):
    data_a = np.random.randint(0, 255, shape)
    data_b = np.zeros_like(data_a)
    chunks = (3,) * data_a.ndim
    group_spec = GroupSpec(
        members={
            'a': ArraySpec.from_array(data_a, attrs={'foo': 100}, chunks=chunks),
            'b': ArraySpec.from_array(data_b, chunks=chunks)
            },
            attrs={})
    group = group_spec.to_zarr(zarr.MemoryStore(), path='test')
    arr_1 = group['a']
    arr_1[:] = data_a
    arr_2 = group['b']

    copy_op = copy_array(arr_1, arr_2, keep_attrs=keep_attrs)
    dask.compute(copy_op, scheduler='threads')
    if keep_attrs:
        assert arr_1.attrs == arr_2.attrs
    else:
        assert arr_2.attrs.asdict() == {}
    assert np.array_equal(arr_2, arr_1)


@pytest.mark.parametrize("shape", ((1000,), (100, 100)))
def test_array_copy_from_path(temp_zarr, shape):
    g = zarr.group(zarr.NestedDirectoryStore(temp_zarr))
    arr_1 = g.create_dataset(name="a", data=np.random.randint(0, 255, shape))
    arr_2 = g.create_dataset(name="b", data=np.zeros(arr_1.shape, dtype=arr_1.dtype))

    copy_op = copy_array(arr_1, arr_2)
    dask.compute(copy_op)
    assert np.array_equal(arr_2, arr_1)


def test_write_blocks_delayed():
    arr = da.random.randint(0, 255, (10, 10, 10), dtype="uint8")
    store = zarr.MemoryStore()
    arr_spec = ArraySpec.from_array(arr, chunks=(2, 2, 2))
    z_arr = arr_spec.to_zarr(store, path="array")
    w_ops = write_blocks_delayed(arr, z_arr)
    result = dask.compute(w_ops)[0]
    assert result == slices_from_chunks(arr.chunks)
    assert_array_equal(np.array(arr), z_arr)


@pytest.mark.parametrize(
    "data, expected",
    [
        (((0, 1), (0, 1)), (0, 0)),
        (((0, 2), (0, 1)), (0, 0)),
        (((1, 2), (0, 1)), (1, 0)),
        (((0, 10), (0, 1)), (0, 0)),
    ],
)
def test_interval_remainder(data, expected):
    assert interval_remainder(*data) == expected

@pytest.mark.parametrize(
    "data, expected",
    [
        (
            (
                slice(None), 
                (0, 10)
            ), 
            slice(0, 10, 1),
        ),
        (
            (
                slice(0, 10, 1), 
                (0, 10)
            ), 
            slice(0, 10, 1),
        ),
                (
            (
                slice(0, 10), 
                (0, 10)
            ), 
            slice(0, 10, 1),
        ),
                (
            (
                slice(None), 
                (9, 10)
            ), 
            slice(9, 10, 1),
        )
    ],
)
def test_resolve_slice(data, expected):
    assert resolve_slice(*data) == expected

@pytest.mark.parametrize(
    "chunks",
    [
    (10,),
    (10,11),
    (10,11,12),
    ],
)
def test_chunksafe_writes(chunks: Tuple[int, ...]):
    store = zarr.MemoryStore()
    array = zarr.open(
        store, path="foo", chunks=chunks, shape=map(lambda v: v * 2, chunks)
    )
    selection = tuple(slice(0, c + 1) for c in chunks)
    valid_data = np.zeros(array.shape) + 1
    setitem(valid_data, array, (slice(None),) * array.ndim)

    slices_resolved = resolve_slices(selection, tuple((0, s) for s in array.shape))
    shape_resolved = tuple(sl.stop - sl.start for sl in slices_resolved)
    invalid_data = np.zeros(shape_resolved) + 2
    
    with pytest.raises(ValueError, match="Planned writes are not chunk-aligned."):
        setitem(invalid_data, array, selection, chunk_safe=True)
