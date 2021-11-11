from xarray.core.common import T
import zarr
from fibsem_tools.io import read, access
import numpy as np
import shutil
import atexit
import os
from fibsem_tools.io.io import read_dask
from fibsem_tools.io.tensorstore import access_precomputed, precomputed_to_dask
from fibsem_tools.io.util import list_files, list_files_parallel, split_path_at_suffix
from pathlib import Path
import os
import tempfile


def _make_local_files(files):
    result = []
    for f in files:
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        Path(f).touch(exist_ok=True)    
        result.append(str(Path(f).absolute()))
    return result


def test_accessing_array_zarr():
    store = tempfile.mkdtemp(suffix='.zarr')
    atexit.register(shutil.rmtree, store)
    data = np.random.randint(0,255, size=(100,), dtype='uint8') 
    z = zarr.open(store, mode='w', shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(read(store)[:], data)

    darr = read_dask(store, chunks=(10,))
    assert (darr.compute() == data).all
    


def test_accessing_array_zarr_n5():
    store = tempfile.mkdtemp(suffix='.n5')
    atexit.register(shutil.rmtree, store)
    data = np.random.randint(0,255, size=(100,), dtype='uint8') 
    z = zarr.open(store, mode='w', shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(read(store)[:], data)

    darr = read_dask(store, chunks=(10,))
    assert (darr.compute() == data).all


def test_accessing_group_zarr():
    store = tempfile.mkdtemp(suffix='.zarr')
    atexit.register(shutil.rmtree, store)
    data = np.zeros(100, dtype='uint8') + 42

    zg = zarr.open(store, mode='w')
    zg['foo'] = data
    assert access(store, mode='a') == zg
    
    zg = access(store, mode='w')
    zg['foo'] = data
    assert zarr.open(store, mode='a') == zg


def test_accessing_group_zarr_n5():
    store = tempfile.mkdtemp(suffix='.n5')
    atexit.register(shutil.rmtree, store)
    data = np.zeros(100, dtype='uint8') + 42
    zg = zarr.open(store, mode='a')
    zg.attrs.update({'foo': 'bar'})
    zg['foo'] = data

    assert dict(access(store, mode='r').attrs) == {'foo': 'bar'}
    assert np.array_equal(access(store, mode='r')['foo'][:], data) 

def test_accessing_precomputed():
    store = tempfile.mkdtemp(suffix='.precomputed')
    atexit.register(shutil.rmtree, store)
    key = 's0'
    data = np.random.randint(0,255,size=(10,10,10), dtype='uint8')
    resolution = [1.0, 2.0, 3.0]
    chunks = (2,) * data.ndim

    arr_out = access_precomputed(store, key, mode='w', array_type='image', dtype='uint8',shape=data.shape, num_channels=1, resolution=resolution, encoding='raw', chunks=chunks)
    arr_out[:] = data.reshape(*data.shape,1)
    
    arr_in = access_precomputed(store, key, mode='r')
    assert np.all(np.array(arr_in[:]) == data.reshape(*data.shape, 1))
    
    darr = precomputed_to_dask(os.path.join(store, key), chunks=(2,2,2))
    assert (darr.compute() ==  data).all


def test_list_files():
    path = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, path)
    fnames = [os.path.join(path, f) for f in [os.path.join('foo', 'foo.txt'), os.path.join('foo', 'bar', 'bar.txt'), os.path.join('foo', 'bar' , 'baz', 'baz.txt')]]
    files_made = _make_local_files(fnames)    
    files_found = list_files(path)
    assert set(files_found) == set(fnames)


def test_list_files_parellel():
    path = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, path)
    fnames = [os.path.join(path, f) for f in [os.path.join('foo', 'foo.txt'), os.path.join('foo', 'bar', 'bar.txt'), os.path.join('foo','bar','baz','baz.txt')]]
    files_made = _make_local_files(fnames) 
    files_found = list_files_parallel(path)
    assert set(files_found) == set(fnames)


def test_path_splitting():
    path = 's3://0/1/2.n5/3/4'
    split = split_path_at_suffix(path, ('.n5',))
    assert(split == ('s3://0/1/2.n5', '3/4', '.n5'))
    
    path = os.path.join('0', '1' , '2.n5', '3' ,'4')
    split = split_path_at_suffix(path, ('.n5',))
    assert(split == (os.path.join('0', '1' , '2.n5'), os.path.join('3', '4'), '.n5'))
    
    path = os.path.join('0', '1' , '2.n5')
    split = split_path_at_suffix(path, ('.n5',))
    assert(split == (os.path.join('0', '1', '2.n5'), '', '.n5'))

def test_n5fsstore():
    with tempfile.TemporaryDirectory() as store_path:
        store = zarr.N5Store(store_path)
        group = zarr.open(store, mode='w')
        arr = group.create_dataset(name='data', shape=(10,10), dtype='int')
        arr[:] = 42

        store2 = zarr.N5FSStore(store_path)
        arr2 = zarr.open(store2, path='data', mode='r')
        np.testing.assert_array_equal(arr[:], arr2[:])

    with tempfile.TemporaryDirectory() as store_path:
        store = zarr.N5FSStore(store_path)
        group = zarr.open(store, mode='w')
        arr = group.create_dataset(name='data', shape=(10,10), dtype='int')
        arr[:] = 42

        store2 = zarr.N5Store(store_path)
        arr2 = zarr.open(store2, path='data', mode='r')
        np.testing.assert_array_equal(arr[:], arr2[:]) 