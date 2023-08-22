import os
from datatree import DataTree
import pytest
from xarray import DataArray
from zarr.storage import FSStore
from pathlib import Path
import zarr
import numpy as np
import itertools
from fibsem_tools.io.core import read_dask, read_xarray
from fibsem_tools.io.multiscale import multiscale_group
from fibsem_tools.io.xr import stt_from_array
from fibsem_tools.io.zarr import (
    DEFAULT_ZARR_STORE,
    DEFAULT_N5_STORE,
    access_n5,
    access_zarr,
    create_dataarray,
    create_datatree,
    get_chunk_keys,
    get_url,
    to_dask,
    to_xarray,
)
from fibsem_tools.metadata.transform import STTransform
import dask.array as da
from xarray.testing import assert_equal


def test_url(temp_zarr):
    store = FSStore(temp_zarr)
    group = zarr.group(store)
    arr = group.create_dataset(name="foo", data=np.arange(10))
    assert get_url(arr) == f"file://{store.path}/foo"


def test_read_xarray(temp_zarr):
    path = "test"
    url = os.path.join(temp_zarr, path)

    data = {
        "s0": stt_from_array(
            np.zeros((10, 10, 10)),
            dims=("z", "y", "x"),
            scales=(1, 2, 3),
            translates=(0, 1, 2),
            units=("nm", "m", "mm"),
            name="data",
        )
    }
    data["s1"] = data["s0"].coarsen({d: 2 for d in data["s0"].dims}).mean()
    data["s1"].name = "data"

    g_spec = multiscale_group(
        arrays=tuple(data.values()),
        metadata_types=["cosem"],
        array_paths=tuple(data.keys()),
        chunks=(64, 64, 64),
    )
    zgroup = g_spec.to_zarr(zarr.NestedDirectoryStore(temp_zarr), path=path)

    for key, value in data.items():
        zgroup[key] = value.data
        zgroup[key].attrs["transform"] = STTransform.from_xarray(value).dict()

    tree_expected = DataTree.from_dict(data, name=path)
    assert_equal(to_xarray(zgroup["s0"]), data["s0"])
    assert_equal(to_xarray(zgroup["s1"]), data["s1"])
    assert tree_expected.equals(to_xarray(zgroup))
    assert tree_expected.equals(read_xarray(url))


@pytest.mark.parametrize("attrs", (None, {"foo": 10}))
@pytest.mark.parametrize("coords", ("auto",))
@pytest.mark.parametrize("use_dask", (True, False))
@pytest.mark.parametrize("name", (None, "foo"))
def test_read_datatree(temp_zarr, attrs, coords, use_dask, name):
    path = "test"
    os.path.join(temp_zarr, path)
    base_data = np.zeros((10, 10, 10))
    store = zarr.NestedDirectoryStore(temp_zarr)
    if attrs is None:
        _attrs = {}
    else:
        _attrs = attrs

    if name is None:
        name_expected = path
    else:
        name_expected = name

    data = {
        "s0": stt_from_array(
            base_data,
            dims=("z", "y", "x"),
            scales=(1, 2, 3),
            translates=(0, 1, 2),
            units=("nm", "m", "mm"),
            name="data",
        )
    }
    data["s1"] = data["s0"].coarsen({d: 2 for d in data["s0"].dims}).mean()
    data["s1"].name = "data"

    g_spec = multiscale_group(
        arrays=tuple(data.values()),
        array_paths=tuple(data.keys()),
        chunks=(64, 64, 64),
        metadata_types=["cosem"],
    )
    g_spec.attrs.update(**_attrs)
    group = g_spec.to_zarr(store, path=path)

    for key, value in data.items():
        group[key] = value.data
        group[key].attrs["transform"] = STTransform.from_xarray(value).dict()

    data_store = create_datatree(
        access_zarr(temp_zarr, path, mode="r"),
        use_dask=use_dask,
        name=name,
        coords=coords,
        attrs=attrs,
    )

    if name is None:
        assert data_store.name == group.basename
    else:
        assert data_store.name == name

    assert (
        all(isinstance(d["data"].data, da.Array) for k, d in data_store.items())
        == use_dask
    )

    # create a datatree directly
    tree_dict = {
        k: DataArray(
            access_zarr(temp_zarr, os.path.join(path, k), mode="r"),
            coords=data[k].coords,
            name="data",
        )
        for k in data.keys()
    }

    tree_expected = DataTree.from_dict(tree_dict, name=name_expected)

    assert tree_expected.equals(data_store)
    if attrs is None:
        assert dict(data_store.attrs) == dict(group.attrs)
    else:
        assert dict(data_store.attrs) == attrs


@pytest.mark.parametrize("attrs", (None, {"foo": 10}))
@pytest.mark.parametrize("coords", ("auto",))
@pytest.mark.parametrize("use_dask", (True, False))
@pytest.mark.parametrize("name", (None, "foo"))
def test_read_dataarray(temp_zarr, attrs, coords, use_dask, name):
    path = "test"
    data = stt_from_array(
        np.zeros((10, 10, 10)),
        dims=("z", "y", "x"),
        scales=(1, 2, 3),
        translates=(0, 1, 2),
        units=("nm", "m", "mm"),
    )

    if attrs is None:
        _attrs = {}
    else:
        _attrs = attrs

    tmp_zarr = access_zarr(
        temp_zarr,
        path,
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        attrs={"transform": STTransform.from_xarray(data).dict(), **_attrs},
    )

    tmp_zarr[:] = data.data
    data_store = create_dataarray(
        access_zarr(temp_zarr, path, mode="r"),
        use_dask=use_dask,
        attrs=attrs,
        coords=coords,
        name=name,
    )

    if name is None:
        assert data_store.name == tmp_zarr.basename
    else:
        assert data_store.name == name

    assert_equal(data_store, data)
    assert isinstance(data_store.data, da.Array) == use_dask

    if attrs is None:
        assert dict(tmp_zarr.attrs) == dict(data_store.attrs)
    else:
        assert dict(data_store.attrs) == attrs


def test_access_array_zarr(temp_zarr):
    data = np.random.randint(0, 255, size=(100,), dtype="uint8")
    z = zarr.open(temp_zarr, mode="w", shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(access_zarr(temp_zarr, "", mode="r")[:], data)


def test_access_array_n5(temp_n5):
    data = np.random.randint(0, 255, size=(100,), dtype="uint8")
    z = zarr.open(temp_n5, mode="w", shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(access_n5(temp_n5, "", mode="r")[:], data)


@pytest.mark.parametrize("accessor", (access_zarr, access_n5))
def test_access_group(temp_zarr, accessor):
    if accessor == access_zarr:
        default_store = DEFAULT_ZARR_STORE
    elif accessor == access_n5:
        default_store = DEFAULT_N5_STORE
    data = np.zeros(100, dtype="uint8") + 42
    path = "foo"
    zg = zarr.open(default_store(temp_zarr), mode="a")
    zg[path] = data
    zg.attrs["bar"] = 10
    assert accessor(temp_zarr, "", mode="a") == zg

    zg = accessor(temp_zarr, "", mode="w", attrs={"bar": 10})
    zg["foo"] = data
    assert zarr.open(default_store(temp_zarr), mode="a") == zg


@pytest.mark.parametrize("chunks", ("auto", (10,)))
def test_dask(temp_zarr, chunks):
    path = "foo"
    data = np.arange(100)
    zarray = access_zarr(temp_zarr, path, mode="w", shape=data.shape, dtype=data.dtype)
    zarray[:] = data
    name_expected = "foo"

    expected = da.from_array(zarray, chunks=chunks, name=name_expected)
    observed = to_dask(zarray, chunks=chunks, name=name_expected)

    assert observed.chunks == expected.chunks
    assert observed.name == expected.name
    assert np.array_equal(observed, data)

    assert np.array_equal(read_dask(get_url(zarray), chunks).compute(), data)


@pytest.mark.parametrize(
    "store_class", (zarr.N5Store, zarr.DirectoryStore, zarr.NestedDirectoryStore)
)
@pytest.mark.parametrize("shape", ((10,), (10, 11, 12)))
def test_chunk_keys(tmp_path: Path, store_class, shape):
    store: zarr.storage.BaseStore = store_class(tmp_path)
    arr_path = "test"
    arr = zarr.create(
        shape=shape, store=store, path=arr_path, chunks=(2,) * len(shape), dtype="uint8"
    )

    dim_sep = arr._dimension_separator
    chunk_idcs = itertools.product(*(range(c_s) for c_s in arr.cdata_shape))
    expected = tuple(
        os.path.join(arr.path, dim_sep.join(map(str, idx))) for idx in chunk_idcs
    )
    observed = tuple(get_chunk_keys(arr))
    assert observed == expected
