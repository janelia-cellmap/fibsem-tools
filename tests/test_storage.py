from xarray.core.common import T
import zarr
from fibsem_tools.io import read, access
import numpy as np
import shutil
from fibsem_tools.io.io import daskify
from fibsem_tools.io.tensorstore import access_precomputed, precomputed_to_dask
from fibsem_tools.io.util import list_files, list_files_parallel, split_path_at_suffix
from pathlib import Path

def _make_local_files(files):
    result = []
    for f in files:
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        Path(f).touch(exist_ok=True)    
        result.append(str(Path(f).absolute()))
    return result


def test_accessing_array_zarr():
    store = 'data/array.zarr'
    data = np.random.randint(0,255, size=(100,), dtype='uint8') 
    z = zarr.open(store, mode='w', shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(read(store)[:], data)

    darr = daskify(store, chunks=(10,))
    assert (darr.compute() == data).all
    shutil.rmtree(store)


def test_accessing_array_zarr_n5():
    store = 'data/array.n5'
    data = np.random.randint(0,255, size=(100,), dtype='uint8') 
    z = zarr.open(store, mode='w', shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(read(store)[:], data)

    darr = daskify(store, chunks=(10,))
    assert (darr.compute() == data).all
    shutil.rmtree(store)


def test_accessing_group_zarr():
    store = 'data/group.zarr'
    data = np.zeros(100, dtype='uint8') + 42

    zg = zarr.open(store, mode='w')
    zg['foo'] = data
    assert access(store, mode='a') == zg
    
    zg = access(store, mode='w')
    zg['foo'] = data
    assert zarr.open(store, mode='a') == zg
    shutil.rmtree(store)

def test_accessing_group_zarr_n5():
    store = 'data/group.n5'
    data = np.zeros(100, dtype='uint8') + 42
    zg = zarr.open(store, mode='w')
    zg['foo'] = data
    assert access(store, mode='a') == zg

    zg = access(store, mode='w')
    zg['foo'] = data
    assert zarr.open(store, mode='a') == zg

    shutil.rmtree(store)


def test_accessing_precomputed():
    store = 'data/array.precomputed/'
    key = 's0'
    data = np.random.randint(0,255,size=(10,10,10), dtype='uint8')
    resolution = [1.0, 2.0, 3.0]
    chunks = (2,) * data.ndim

    arr_out = access_precomputed(store, key, mode='w', array_type='image', dtype='uint8',shape=data.shape, num_channels=1, resolution=resolution, encoding='raw', chunks=chunks)
    arr_out[:] = data.reshape(*data.shape,1)
    
    arr_in = access_precomputed(store, key, mode='r')
    assert np.all(np.array(arr_in[:]) == data.reshape(*data.shape, 1))
    
    darr = precomputed_to_dask(store, key, chunks=(2,2,2,1))
    assert (darr.compute() ==  data).all
    shutil.rmtree('data/array.precomputed')

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

def test_path_splitting():
    path = 's3://0/1/2.n5/3/4'
    split = split_path_at_suffix(path, ('.n5',))
    assert(split == ('s3://0/1/2.n5', '3/4', '.n5'))
    
    path = '/0/1/2.n5/3/4'
    split = split_path_at_suffix(path, ('.n5',))
    assert(split == ('/0/1/2.n5', '3/4', '.n5'))
    
    path = '/0/1/2.n5'
    split = split_path_at_suffix(path, ('.n5',))
    assert(split == ('/0/1/2.n5', '', '.n5'))