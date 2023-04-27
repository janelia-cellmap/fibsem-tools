from datatree import DataTree
import numpy as np
import pytest
import dask.array as da
from fibsem_tools.io.core import (
    access,
    create_dataarray,
    create_datatree,
    read,
)
from fibsem_tools.io.multiscale import multiscale_group
from fibsem_tools.io.xr import stt_coord, stt_from_array, flip
import os
from xarray.testing import assert_equal
from fibsem_tools.metadata.transform import STTransform


@pytest.mark.parametrize("length", (1, 10, 100))
@pytest.mark.parametrize("dim", ("a", "x"))
@pytest.mark.parametrize("scale", (1, 2.5, 3))
@pytest.mark.parametrize("unit", ("nm", "km"))
@pytest.mark.parametrize("translate", (0, -5, 5.5))
def test_stt_coord(length: int, dim: str, translate: float, scale: float, unit: str):
    coordvar = stt_coord(length, dim, scale, translate, unit)
    assert len(coordvar) == length
    assert coordvar.dims == (dim,)
    if len(coordvar) > 1:
        assert coordvar[1] - coordvar[0] == scale
    assert coordvar[0] == translate
    assert coordvar.units == unit


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dims=("a",), scales=(1,), translates=(0,), units=("nm")),
        dict(dims=("a", "b"), scales=(0, 1), translates=(0, 3), units=("nm", "m")),
    ],
)
def test_stt_from_array(kwargs):
    ndim = len(kwargs["dims"])
    data = np.zeros((10,) * ndim)
    xr = stt_from_array(data, **kwargs)
    assert xr.shape == data.shape
    assert xr.dims == kwargs["dims"]
    for idx, coordvar in enumerate(xr.coords.values()):
        assert coordvar.dims == (kwargs["dims"][idx],)
        if len(coordvar) > 1:
            assert coordvar[1] - coordvar[0] == kwargs["scales"][idx]
        assert coordvar[0] == kwargs["translates"][idx]
        assert coordvar.units == kwargs["units"][idx]


@pytest.mark.parametrize("flip_dims", (("a",), ("a", "b"), ("a", "b", "c")))
def test_flip(flip_dims):
    all_dims = ("a", "b", "c")
    ndim = len(all_dims)
    data = stt_from_array(
        np.random.randint(0, 255, (3,) * ndim),
        dims=all_dims,
        scales=(1,) * ndim,
        translates=(0,) * ndim,
        units=("mm",) * ndim,
    )
    test_selector = ()
    for dim in all_dims:
        if dim in flip_dims:
            test_selector += (slice(None, None, -1),)
        else:
            test_selector += (slice(None),)

    assert np.array_equal(flip(data, flip_dims).data, data.data[test_selector])


@pytest.mark.parametrize("use_dask", (True, False))
@pytest.mark.parametrize("coords", ("auto",))
@pytest.mark.parametrize("keep_attrs", (True, False))
def test_read_dataarray(temp_zarr, keep_attrs, coords, use_dask):
    url = os.path.join(temp_zarr, "test")
    data = stt_from_array(
        np.zeros((10, 10, 10)),
        dims=("z", "y", "x"),
        scales=(1, 2, 3),
        translates=(0, 1, 2),
        units=("nm", "m", "mm"),
    )
    tmp_zarr = access(
        url,
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        attrs={"transform": STTransform.fromDataArray(data).dict(), "foo": "bar"},
    )

    tmp_zarr[:] = data.data
    data_store = create_dataarray(
        read(url), keep_attrs=keep_attrs, use_dask=use_dask, coords=coords
    )
    assert_equal(data_store, data)
    assert isinstance(data_store.data, da.Array) == use_dask
    assert (dict(tmp_zarr.attrs) == dict(data_store.attrs)) == keep_attrs


def test_read_datatree(temp_zarr):
    url = os.path.join(temp_zarr, "test")
    data = {
        "s0": stt_from_array(
            np.zeros((10, 10, 10)),
            dims=("z", "y", "x"),
            scales=(1, 2, 3),
            translates=(0, 1, 2),
            units=("nm", "m", "mm"),
            name="s0",
        )
    }
    data["s1"] = data["s0"].coarsen({d: 2 for d in data["s0"].dims}).mean()
    data["s1"].name = "s1"
    data_tree = DataTree.from_dict(data, name="test")
    tmp_zarr = multiscale_group(
        url,
        tuple(data.values()),
        tuple(data.keys()),
        chunks=((64, 64, 64), (64, 64, 64)),
        metadata_types=["cosem"],
    )
    for key, value in data.items():
        tmp_zarr[key] = value.data
        tmp_zarr[key].attrs["transform"] = STTransform.fromDataArray(value).dict()

    data_store = create_datatree(read(url))
    assert data_store == data_tree
