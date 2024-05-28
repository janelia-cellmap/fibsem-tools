from __future__ import annotations

from typing import TYPE_CHECKING

from fibsem_tools.io.zarr import get_url

if TYPE_CHECKING:
    from typing import Literal

import dask.array as da
import numpy as np
import pytest
import zarr
from fibsem_tools.io.n5 import access, to_dask


def test_access_array(tmp_n5: str) -> None:
    data = np.random.randint(0, 255, size=(100,), dtype="uint8")
    z = zarr.open(tmp_n5, mode="w", shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(access(tmp_n5, "", mode="r")[:], data)


def test_access_group(tmp_zarr: str) -> None:
    data = np.zeros(100, dtype="uint8") + 42
    path = "foo"
    zg = zarr.open(zarr.N5FSStore(tmp_zarr), mode="a")
    zg[path] = data
    zg.attrs["bar"] = 10
    assert access(tmp_zarr, "", mode="a") == zg

    zg = access(tmp_zarr, "", mode="w", attrs={"bar": 10})
    zg["foo"] = data
    assert zarr.open(zarr.N5FSStore(tmp_zarr), mode="a") == zg


@pytest.mark.parametrize("chunks", ("auto", (10,)))
def test_dask(tmp_n5: str, chunks: Literal["auto"] | tuple[int, ...]) -> None:
    path = "foo"
    data = np.arange(100)
    zarray = access(tmp_n5, path, mode="w", shape=data.shape, dtype=data.dtype)
    zarray[:] = data
    name_expected = "foo"

    expected = da.from_array(zarray, chunks=chunks, name=name_expected)
    observed = to_dask(zarray, chunks=chunks, name=name_expected)

    assert observed.chunks == expected.chunks
    assert observed.name == expected.name
    assert np.array_equal(observed, data)
