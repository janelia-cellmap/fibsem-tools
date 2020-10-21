from xarray.core.common import T
import zarr
from fst.io import read, access
import numpy as np
import shutil

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


def test_open_group_zarr():
    store = '/data/group.zarr'
    zg = zarr.open(store, mode='w')
    zg['foo'] = np.zeros(100, dtype='uint8') + 42
    assert access(store, mode='w') == zg
    shutil.rmtree(store)


def test_open_group_zarr_n5():
    store = '/data/group.n5'
    zg = zarr.open(store, mode='w')
    zg['foo'] = np.zeros(100, dtype='uint8') + 42
    assert access(store, mode='w') == zg
    shutil.rmtree(store)