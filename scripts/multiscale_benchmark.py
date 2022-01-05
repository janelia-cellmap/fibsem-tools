from fibsem_tools.io.multiscale import Multiscales
from xarray_multiscale import multiscale
from fibsem_tools.io import read, read_xarray
from fibsem_tools.io.dask import ensure_minimum_chunksize
import numpy as np
import dask
from tlz import get
from distributed import Client, performance_report
from dask_janelia import get_cluster
import os

num_workers = 4
tpw = 2
chunk_locking = True
name = f'lsf_nw-{num_workers}_tpw-{tpw}_chunk-locking-{chunk_locking}'
levels = list(range(1,6))
crop = (slice(8192),) * 3

def reducer(v, **kwargs):
    return np.mean(v, dtype='float32', **kwargs)

source_path = '/nrs/flyem/bench/Z0720_07m_BR.n5/render/Sec30/v1_acquire_trimmed_align___20210413_194018/s0'
target_path = '/nrs/flyem/bench/Z0720_07m_BR.n5/test_dask_down/'

store_chunks = read(source_path, storage_options={'normalize_keys': False}).chunks
read_chunks=(1024,) * 3

data = read_xarray(source_path, storage_options={'normalize_keys': False}, chunks=read_chunks, name='test_data')[crop]

multi = get(levels, multiscale(data, reducer, (2,2,2)))

if not chunk_locking:
    for m in multi:
        m.data = ensure_minimum_chunksize(m.data, store_chunks)

    
multi_store = Multiscales(name, {f's{l}' : m for l,m in zip(levels, multi)})

if __name__ == '__main__':
    with get_cluster(threads_per_worker=tpw) as cluster, Client(cluster) as cl:
        print(cl.cluster.dashboard_link)
        cl.cluster.scale(num_workers)
        cl.wait_for_workers(num_workers)
        with performance_report(filename=os.path.join(target_path, f'{name}_report.html')):
            store_group, store_arrays, storage_op = multi_store.store(target_path, locking=chunk_locking, client=cl, mode='w')
            result = cl.compute(dask.delayed(storage_op), sync=True)