import os
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union
from datatree import DataTree
import pytest
from xarray import DataArray
from zarr.storage import FSStore
from pathlib import Path
import zarr
import numpy as np
import itertools
from fibsem_tools.io.core import read_dask, read_xarray
from fibsem_tools.io.multiscale import model_multiscale_group
from fibsem_tools.io.xr import stt_array
from fibsem_tools.io.zarr import (
    DEFAULT_ZARR_STORE,
    DEFAULT_N5_STORE,
    access_n5,
    access_zarr,
    create_dataarray,
    create_datatree,
    chunk_keys,
    get_url,
    to_dask,
    to_xarray,
    array_from_dask,
)
from fibsem_tools.metadata.transform import STTransform
import dask.array as da
from xarray.testing import assert_equal
from fibsem_tools.io.zarr import parse_url


def test_url(tmp_zarr: str) -> None:
    store = FSStore(tmp_zarr)
    group = zarr.group(store)
    arr = group.create_dataset(name="foo", data=np.arange(10))
    assert get_url(arr) == f"file://{store.path}/foo"


def test_read_xarray(tmp_zarr: str) -> None:
    path = "test"
    url = os.path.join(tmp_zarr, path)

    data = {
        "s0": stt_array(
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

    g_spec = model_multiscale_group(
        arrays=data,
        metadata_type="ome-ngff",
        chunks=(64, 64, 64),
    )
    zgroup = g_spec.to_zarr(zarr.NestedDirectoryStore(tmp_zarr), path=path)

    for key, value in data.items():
        zgroup[key] = value.data
        zgroup[key].attrs["transform"] = STTransform.from_xarray(value).model_dump()

    tree_expected = DataTree.from_dict(data, name=path)
    assert_equal(to_xarray(zgroup["s0"]), data["s0"])
    assert_equal(to_xarray(zgroup["s1"]), data["s1"])
    assert tree_expected.equals(to_xarray(zgroup))
    assert tree_expected.equals(read_xarray(url))


@pytest.mark.parametrize("attrs", (None, {"foo": 10}))
@pytest.mark.parametrize("coords", ("auto",))
@pytest.mark.parametrize("use_dask", (True, False))
@pytest.mark.parametrize("name", (None, "foo"))
@pytest.mark.parametrize("path", ("a", "a/b"))
def test_read_datatree(
    tmp_zarr: str,
    attrs: Optional[Dict[str, Any]],
    coords: str,
    use_dask: bool,
    name: Optional[str],
    path: str,
) -> None:
    base_data = np.zeros((10, 10, 10))
    store = zarr.NestedDirectoryStore(tmp_zarr)
    if attrs is None:
        _attrs = {}
    else:
        _attrs = attrs

    if name is None:
        name_expected = path.split("/")[-1]
    else:
        name_expected = name

    data = {
        "s0": stt_array(
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

    g_spec = model_multiscale_group(
        arrays=data,
        chunks=((64, 64, 64),) * len(data),
        metadata_type="ome-ngff",
    )
    group = g_spec.to_zarr(store, path=path)
    group.attrs.update(**_attrs)

    for key, value in data.items():
        group[key] = value.data
        # group[key].attrs["transform"] = STTransform.from_xarray(value).model_dump()

    data_store = create_datatree(
        access_zarr(tmp_zarr, path, mode="r"),
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
            access_zarr(tmp_zarr, os.path.join(path, k), mode="r"),
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
def test_read_dataarray(
    tmp_zarr: str,
    attrs: Optional[Dict[str, Any]],
    coords: str,
    use_dask: bool,
    name: Optional[str],
) -> None:
    path = "test"
    data = stt_array(
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
        tmp_zarr,
        path,
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        attrs={"transform": STTransform.from_xarray(data).model_dump(), **_attrs},
    )

    tmp_zarr[:] = data.data
    data_store = create_dataarray(
        tmp_zarr,
        use_dask=use_dask,
        attrs=attrs,
        coords=coords,
        name=name,
    )

    if name is None:
        assert data_store.name == tmp_zarr.basename
        if use_dask:
            assert data_store.data.name.startswith(get_url(tmp_zarr))
    else:
        assert data_store.name == name

    assert_equal(data_store, data)
    assert isinstance(data_store.data, da.Array) == use_dask

    if attrs is None:
        assert dict(tmp_zarr.attrs) == dict(data_store.attrs)
    else:
        assert dict(data_store.attrs) == attrs


def test_access_array_zarr(tmp_zarr: str) -> None:
    data = np.random.randint(0, 255, size=(100,), dtype="uint8")
    z = zarr.open(tmp_zarr, mode="w", shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(access_zarr(tmp_zarr, "", mode="r")[:], data)


def test_access_array_n5(tmp_n5: str) -> None:
    data = np.random.randint(0, 255, size=(100,), dtype="uint8")
    z = zarr.open(tmp_n5, mode="w", shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(access_n5(tmp_n5, "", mode="r")[:], data)


@pytest.mark.parametrize("accessor", (access_zarr, access_n5))
def test_access_group(tmp_zarr: str, accessor: Callable[[Any], None]) -> None:
    if accessor == access_zarr:
        default_store = DEFAULT_ZARR_STORE
    elif accessor == access_n5:
        default_store = DEFAULT_N5_STORE
    data = np.zeros(100, dtype="uint8") + 42
    path = "foo"
    zg = zarr.open(default_store(tmp_zarr), mode="a")
    zg[path] = data
    zg.attrs["bar"] = 10
    assert accessor(tmp_zarr, "", mode="a") == zg

    zg = accessor(tmp_zarr, "", mode="w", attrs={"bar": 10})
    zg["foo"] = data
    assert zarr.open(default_store(tmp_zarr), mode="a") == zg


@pytest.mark.parametrize("chunks", ("auto", (10,)))
def test_dask(tmp_zarr: str, chunks: Union[Literal["auto"], Tuple[int, ...]]) -> None:
    path = "foo"
    data = np.arange(100)
    zarray = access_zarr(tmp_zarr, path, mode="w", shape=data.shape, dtype=data.dtype)
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
def test_chunk_keys(
    tmp_path: Path,
    store_class: Union[zarr.N5Store, zarr.DirectoryStore, zarr.NestedDirectoryStore],
    shape: Tuple[int, ...],
) -> None:
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
    observed = tuple(chunk_keys(arr))
    assert observed == expected


@pytest.mark.parametrize("inline_array", (True, False))
def test_zarr_array_from_dask(inline_array: bool) -> None:
    store = zarr.MemoryStore()
    zarray = zarr.open(store, shape=(10, 10))
    darray = da.from_array(zarray, inline_array=inline_array)
    assert array_from_dask(darray) == zarray


@pytest.mark.parametrize(
    "url, expected",
    [
        ("foo.zarr", ("foo.zarr", "")),
        ("/foo/foo.zarr", ("/foo/foo.zarr", "")),
        ("/foo/foo.zarr/bar", ("/foo/foo.zarr", "bar")),
        ("/foo/foo.zarr/bar/baz", ("/foo/foo.zarr", "bar/baz")),
    ],
)
def test_parse_url(url: str, expected: str):
    assert parse_url(url) == expected


@pytest.mark.parametrize("url", ["foo", "foo/bar/baz", "foo/bar/b.zarraz"])
def test_parse_url_no_zarr(url: str):
    with pytest.raises(ValueError, match="None of the parts of the url"):
        parse_url(url)


@pytest.mark.parametrize("url", ["foo.zarr/baz.zarr", "foo.zarr/bar/baz.zarr"])
def test_parse_url_too_much_zarr(url: str):
    with pytest.raises(ValueError, match="Too many parts of the url"):
        parse_url(url)
