from typing import Any, Tuple, List
import dask
import distributed
import dask.array as da
import numpy as np
from dask.array.core import slices_from_chunks, store_chunk


def _blocks(self, index, key_array):
    """
    This only exists until a performance issue in the dask.array.block is sorted out
    """
    from numbers import Number
    from dask.array.slicing import normalize_index
    from dask.base import tokenize

    from itertools import product
    from dask.highlevelgraph import HighLevelGraph
    from dask.array import Array

    if not isinstance(index, tuple):
        index = (index,)
    if sum(isinstance(ind, (np.ndarray, list)) for ind in index) > 1:
        raise ValueError("Can only slice with a single list")
    if any(ind is None for ind in index):
        raise ValueError("Slicing with np.newaxis or None is not supported")
    index = normalize_index(index, self.numblocks)
    index = tuple(slice(k, k + 1) if isinstance(k, Number) else k for k in index)

    name = "blocks-" + tokenize(self, index)

    new_keys = key_array[index]

    chunks = tuple(tuple(np.array(c)[i].tolist()) for c, i in zip(self.chunks, index))

    keys = product(*(range(len(c)) for c in chunks))

    layer = {(name,) + key: tuple(new_keys[key].tolist()) for key in keys}

    graph = HighLevelGraph.from_collections(name, layer, dependencies=[self])
    return Array(graph, name, chunks, meta=self)


def sequential_rechunk(
    source: Any,
    target: Any,
    slab_size: Tuple[int],
    intermediate_chunks: Tuple[int],
    client: distributed.Client,
    num_workers: int,
) -> List[None]:
    """
    Load slabs of an array into local memory, then create a dask array and rechunk that dask array, then store into
    chunked array storage.
    """
    results = []
    slices = slices_from_chunks(source.rechunk(slab_size).chunks)

    for sl in slices:
        arr_in = source[sl].compute(scheduler="threads")
        darr_in = da.from_array(arr_in, chunks=intermediate_chunks)
        store_op = da.store(darr_in, target, regions=sl, compute=False, lock=None)
        client.cluster.scale(num_workers)
        results.extend(client.compute(store_op).result())
        client.cluster.scale(0)
    return results


def write_blocks(source, target, region, lock=None, return_stored=False):
    """
    For each chunk in `source`, write that data to `target`
    """

    storage_op = []
    key_array = np.array(source.__dask_keys__(), dtype=object)
    slices = slices_from_chunks(source.chunks)
    if region:
        slices = [fuse_slice(region, slc) for slc in slices]
    for lidx, aidx in enumerate(np.ndindex(tuple(map(len, source.chunks)))):
        region = slices[lidx]
        source_block = _blocks(source, aidx, key_array)
        storage_op.append(
            dask.delayed(store_chunk)(source_block, target, region, lock, return_stored)
        )
    return storage_op


def store_blocks(sources, targets, regions=None) -> List[List[dask.delayed]]:
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
        result.append(
            write_blocks(source, target, region, lock=None, return_stored=False)
        )
    return result
