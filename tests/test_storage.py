from xarray.core.common import T
import zarr
from fibsem_tools.io import read, access
import numpy as np
import shutil
from fibsem_tools.io.util import list_files, list_files_parallel
from pathlib import Path

def _make_local_files(files):
    result = []
    for f in files:
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        Path(f).touch(exist_ok=True)    
        result.append(str(Path(f).absolute()))
    return result

def test_open_array_zarr():
    store = 'data/array.zarr'
    data = np.zeros(100, dtype='uint8') + 42
    z = zarr.open(store, mode='w', shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(read(store)[:], data)
    shutil.rmtree(store)


def test_open_array_zarr_n5():
    store = 'data/array.n5'
    data = np.zeros(100, dtype='uint8') + 42
    z = zarr.open(store, mode='w', shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(read(store)[:], data)
    shutil.rmtree(store)


def test_group_creation_zarr():
    store = '/data/group.zarr'
    data = np.zeros(100, dtype='uint8') + 42

    zg = zarr.open(store, mode='w')
    zg['foo'] = data
    assert access(store, mode='a') == zg
    
    zg = access(store, mode='w')
    zg['foo'] = data
    assert zarr.open(store, mode='a') == zg
    shutil.rmtree(store)

def test_group_creation_zarr_n5():
    store = '/data/group.n5'
    data = np.zeros(100, dtype='uint8') + 42
    zg = zarr.open(store, mode='w')
    zg['foo'] = data
    assert access(store, mode='a') == zg

    zg = access(store, mode='w')
    zg['foo'] = data
    assert zarr.open(store, mode='a') == zg

    shutil.rmtree(store)


def test_list_files():
    fnames = ['./foo/foo.txt', './foo/bar/bar.txt', './foo/bar/baz/baz.txt']
    files_made = _make_local_files(fnames)    
    files_found = list_files('./foo')
    assert set(files_found) == set(fnames)

    shutil.rmtree('foo')

def test_list_files_parellel():
    fnames = ['./foo/foo.txt', './foo/bar/bar.txt', './foo/bar/baz/baz.txt']
    files_made = _make_local_files(fnames) 
    files_found = list_files_parallel('./foo')
    assert set(files_found) == set(fnames)

    shutil.rmtree('foo')