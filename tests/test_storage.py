import atexit
import os
import shutil
import tempfile
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import zarr

from fibsem_tools.io.dask import store_blocks
from fibsem_tools.io.core import create_group, access, read
from fibsem_tools.io.util import list_files, list_files_parallel, split_by_suffix
from fibsem_tools.io.zarr import delete_zbranch


def _make_local_files(files):
    result = []
    for f in files:
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        Path(f).touch(exist_ok=True)
        result.append(str(Path(f).absolute()))
    return result


def test_list_files(temp_dir):
    fnames = [
        os.path.join(temp_dir, f)
        for f in [
            os.path.join("foo", "foo.txt"),
            os.path.join("foo", "bar", "bar.txt"),
            os.path.join("foo", "bar", "baz", "baz.txt"),
        ]
    ]
    _make_local_files(fnames)
    files_found = list_files(temp_dir)
    assert set(files_found) == set(fnames)


def test_list_files_parellel(temp_dir):
    fnames = [
        os.path.join(temp_dir, f)
        for f in [
            os.path.join("foo", "foo.txt"),
            os.path.join("foo", "bar", "bar.txt"),
            os.path.join("foo", "bar", "baz", "baz.txt"),
        ]
    ]
    _make_local_files(fnames)
    files_found = list_files_parallel(temp_dir)
    assert set(files_found) == set(fnames)


def test_path_splitting():
    path = "s3://0/1/2.n5/3/4"
    split = split_by_suffix(path, (".n5",))
    assert split == ("s3://0/1/2.n5", "3/4", ".n5")

    path = os.path.join("0", "1", "2.n5", "3", "4")
    split = split_by_suffix(path, (".n5",))
    assert split == (os.path.join("0", "1", "2.n5"), os.path.join("3", "4"), ".n5")

    path = os.path.join("0", "1", "2.n5")
    split = split_by_suffix(path, (".n5",))
    assert split == (os.path.join("0", "1", "2.n5"), "", ".n5")


def test_store_blocks(temp_zarr):
    data = da.arange(256).reshape(16, 16).rechunk((4, 4))
    z = zarr.open(temp_zarr, mode="w", shape=data.shape, chunks=data.chunksize)
    dask.delayed(store_blocks(data, z)).compute()
    assert np.array_equal(read(temp_zarr)[:], data.compute())


def test_group_initialization():
    store_path = tempfile.mkdtemp(suffix=".zarr")
    atexit.register(shutil.rmtree, store_path)
    data = {"foo": np.arange(10), "bar": np.arange(20)}
    a_attrs = {"foo": {"a": 10}, "bar": {"b": 15}}
    g_attrs = {"bla": "bla"}
    chunks = ((2,), (2,))
    group = create_group(
        store_path,
        data.values(),
        data.keys(),
        chunks=chunks,
        group_attrs=g_attrs,
        array_attrs=a_attrs.values(),
    )
    assert g_attrs == dict(group.attrs)
    for d in data:
        assert data[d].shape == group[d].shape
        assert data[d].dtype == group[d].dtype
        assert a_attrs[d] == dict(group[d].attrs)


def test_deletion(temp_zarr):
    existing = access(f"{temp_zarr}/bar", mode="w", shape=(10,), chunks=(1,))
    existing[:] = 10
    keys = tuple(existing.chunk_store.keys())

    for key in keys:
        assert key in existing.chunk_store

    # delete
    delete_zbranch(existing)

    for key in keys:
        if existing.name in key:
            assert key not in existing.chunk_store
