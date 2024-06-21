from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pytest
import zarr
from dask.array.core import slices_from_chunks
from numpy.testing import assert_array_equal
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from fibsem_tools.chunk import (
    resolve_slices,
)
from fibsem_tools.io.core import read
from fibsem_tools.io.dask import (
    copy_array,
    setitem,
    store_blocks,
    write_blocks_delayed,
)


@pytest.mark.parametrize("keep_attrs", [True, False])
@pytest.mark.parametrize("shape", [(10,), (10, 10)])
def test_array_copy_from_array(shape, keep_attrs) -> None:
    data_a = np.random.randint(0, 255, shape)
    data_b = np.zeros_like(data_a)
    chunks = (3,) * data_a.ndim
    group_spec = GroupSpec(
        members={
            "a": ArraySpec.from_array(data_a, attributes={"foo": 100}, chunks=chunks),
            "b": ArraySpec.from_array(data_b, chunks=chunks),
        }
    )
    group = group_spec.to_zarr(zarr.MemoryStore(), path="test")
    arr_1 = group["a"]
    arr_1[:] = data_a
    arr_2 = group["b"]

    copy_op = copy_array(arr_1, arr_2, keep_attrs=keep_attrs)
    dask.compute(copy_op, scheduler="threads")
    if keep_attrs:
        assert arr_1.attrs == arr_2.attrs
    else:
        assert arr_2.attrs.asdict() == {}
    assert np.array_equal(arr_2, arr_1)


@pytest.mark.parametrize("shape", [(1000,), (100, 100)])
def test_array_copy_from_path(tmp_zarr, shape) -> None:
    g = zarr.group(zarr.NestedDirectoryStore(tmp_zarr))
    arr_1 = g.create_dataset(name="a", data=np.random.randint(0, 255, shape))
    arr_2 = g.create_dataset(name="b", data=np.zeros(arr_1.shape, dtype=arr_1.dtype))

    copy_op = copy_array(arr_1, arr_2)
    dask.compute(copy_op)
    assert np.array_equal(arr_2, arr_1)


def test_write_blocks_delayed() -> None:
    arr = da.random.randint(0, 255, (10, 10, 10), dtype="uint8")
    store = zarr.MemoryStore()
    arr_spec = ArraySpec.from_array(arr, chunks=(2, 2, 2))
    z_arr = arr_spec.to_zarr(store, path="array")
    w_ops = write_blocks_delayed(arr, z_arr)
    result = dask.compute(w_ops)[0]
    assert result == slices_from_chunks(arr.chunks)
    assert_array_equal(np.array(arr), z_arr)


@pytest.mark.parametrize(
    "chunks",
    [
        (10,),
        (10, 11),
        (10, 11, 12),
    ],
)
def test_chunksafe_writes(chunks: tuple[int, ...]) -> None:
    store = zarr.MemoryStore()
    array = zarr.open(
        store, path="foo", chunks=chunks, shape=tuple(v * 2 for v in chunks)
    )
    selection = tuple(slice(0, c + 1) for c in chunks)
    valid_data = np.zeros(array.shape) + 1
    setitem(valid_data, array, (slice(None),) * array.ndim)

    slices_resolved = resolve_slices(selection, tuple((0, s) for s in array.shape))
    shape_resolved = tuple(sl.stop - sl.start for sl in slices_resolved)
    invalid_data = np.zeros(shape_resolved) + 2

    with pytest.raises(ValueError, match="Planned writes are not chunk-aligned."):
        setitem(invalid_data, array, selection, chunk_safe=True)


def test_store_blocks(tmp_zarr) -> None:
    data = da.arange(256).reshape(16, 16).rechunk((4, 4))
    z = zarr.open(tmp_zarr, mode="w", shape=data.shape, chunks=data.chunksize)
    dask.delayed(store_blocks(data, z)).compute()
    assert np.array_equal(read(tmp_zarr)[:], data.compute())
