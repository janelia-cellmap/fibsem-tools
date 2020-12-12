from xarray.core.common import T
import zarr
from fibsem_tools.io import read, access
import numpy as np
import shutil
from fibsem_tools.io import fwalk, fwalk_parallel
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


def test_fwalk():
    from pathlib import Path
    import shutil

    fnames = ['./foo/foo.txt', './foo/bar/bar.txt', './foo/bar/baz/baz.txt']
    files_made = _make_local_files(fnames)    
    files_found = fwalk('./foo')
    assert set(files_found) == set(files_made)

    shutil.rmtree('foo')

def test_fwalk_parellel():
    from pathlib import Path
    import shutil

    fnames = ['foo/foo.txt', 'foo/bar/bar.txt', 'foo/bar/baz/baz.txt']
    files_made = _make_local_files(fnames) 
    files_found = fwalk_parallel(['./foo'])
    assert set(files_found) == set(files_made)

    shutil.rmtree('foo')