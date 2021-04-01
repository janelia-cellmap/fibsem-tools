from typing import Any, Tuple, List
import dask
import distributed
import dask.array as da
import numpy as np
import random, time
def sequential_rechunk(source: Any, target: Any, slab_size: Tuple[int], intermediate_chunks: Tuple[int], client: distributed.Client, num_workers: int) -> List[None]:
    """
    Load slabs of an array into local memory, then create a dask array and rechunk that dask array, then store into 
    chunked array storage.
    """
    results = []
    slices = da.core.slices_from_chunks(source.rechunk(slab_size).chunks)
    
    for sl in slices:
        arr_in = source[sl].compute(scheduler='threads')
        darr_in = da.from_array(arr_in, chunks=intermediate_chunks)
        store_op = da.store(darr_in,target, regions=sl, compute=False, lock=None)
        client.cluster.scale(num_workers)
        results.extend(client.compute(store_op).result())
        client.cluster.scale(0)
    return results


def store_block(source, target, region, block_info=None, retries: int=5, backoff_seconds: float=1.0):
    backoff_seconds = 1.0
    retries_used = 0
    chunk_origin = block_info[0]["array-location"]
    slices = tuple(slice(start, stop) for start, stop in chunk_origin)
    if region:
        slices = da.optimization.fuse_slice(region, slices)
    while True: 
        try:
            target[slices] = source
            return np.expand_dims(0, tuple(range(source.ndim)))
        except:
            if retries_used == retries:
                return np.expand_dims(1, tuple(range(source.ndim)))
            else:
                sleep_duration = backoff_seconds * 2  ** retries_used + random.uniform(0, 1)
                time.sleep(sleep_duration)
                retries_used += 1 



def store_blocks(sources, targets, regions=None, retries: int=5, backoff_seconds: float=1.0):
    result = []
    
    if isinstance(sources, dask.array.core.Array):
        sources = [sources]
        targets = [targets]
    
    
    if len(sources) != len(targets):
        raise ValueError(
            "Different number of sources [%d] and targets [%d]"
            % (len(sources), len(targets))
        )

    if isinstance(regions, tuple) or regions is None:
        regions = [regions]

    if len(sources) > 1 and len(regions) == 1:
        regions *= len(sources)

    if len(sources) != len(regions):
        raise ValueError(
            "Different number of sources [%d] and targets [%d] than regions [%d]"
            % (len(sources), len(targets), len(regions))
        )
            
    for source, target, region in zip(sources, targets, regions):
        out_chunks = tuple((1,) * len(c) for c in source.chunks)
        result.append(
            da.map_blocks(store_block, source, target, region, chunks=out_chunks, retries=retries, backoff_seconds=backoff_seconds, dtype="int64")
        )
    return result