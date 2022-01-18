from typing import Any, Callable, Literal, Sequence, Tuple, List
import dask
import distributed
import dask.array as da
import numpy as np
from numpy.typing import NDArray
from dask.array.core import slices_from_chunks
import backoff
from dask.array.optimization import fuse_slice
from typing import Any, Tuple, Optional
from aiohttp import ServerDisconnectedError
from dask.utils import is_arraylike
from dask.optimization import fuse
from dask.delayed import Delayed
from dask.core import flatten
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph


def fuse_delayed(tasks: dask.delayed) -> dask.delayed:
    """
    Apply task fusion optimization to tasks. Useful (or even required)
    because dask.delayed optimization doesn't do this step.
    """
    dsk_fused, deps = fuse(dask.utils.ensure_dict(tasks.dask))
    fused = Delayed(tasks._key, dsk_fused)
    return fused


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


@backoff.on_exception(backoff.expo, (ServerDisconnectedError, OSError))
def store_chunk(x: NDArray[Any], out: Any, index: Tuple[slice, ...]) -> Literal[0]:
    """
    A function inserted in a Dask graph for storing a chunk.

    Parameters
    ----------
    x: array-like
        An array (potentially a NumPy one)
    out: array-like
        Where to store results to.
    index: slice-like
        Where to store result from ``x`` in ``out``.

    Examples
    --------

    >>> a = np.ones((5, 6))
    >>> b = np.empty(a.shape)
    >>> load_store_chunk(a, b, (slice(None), slice(None)), False, False, False)
    """

    if is_arraylike(x):
        out[index] = x
    else:
        out[index] = np.asanyarray(x)

    return 0


def ndwrapper(func: Callable[[Any], Any], ndim: int, *args: Any, **kwargs: Any):
    """
    Wrap the result of `func` in a rank-`ndim` numpy array
    """
    return np.array([func(*args, **kwargs)]).reshape((1,) * ndim)


def write_blocks(source, target, region: Optional[Tuple[slice, ...]]) -> da.Array:
    """
    Return a dask array with where each chunk contains the result of writing
    each chunk of `source` to `target`.
    """

    slices = slices_from_chunks(source.chunks)
    if region:
        slices = [fuse_slice(region, slc) for slc in slices]

    source_name = "store-source-" + tokenize(source)
    store_name = "store-" + tokenize(source)

    layers = {source_name: source.__dask_graph__()}
    deps = {source_name: set()}

    dsk = {}
    chunks = tuple((1,) * s for s in source.blocks.shape)

    for slice, key in zip(slices, flatten(source.__dask_keys__())):
        dsk[(store_name,) + key[1:]] = (
            ndwrapper,
            store_chunk,
            source.ndim,
            key,
            target,
            slice,
        )

    layers[store_name] = dsk
    deps[store_name] = {source_name}
    store_dsk = HighLevelGraph(layers, deps)

    return da.Array(
        store_dsk, store_name, shape=source.blocks.shape, chunks=chunks, dtype=int
    )


def store_blocks(sources, targets, regions: Optional[slice] = None) -> List[da.Array]:
    """
    Write dask array(s) to sliceable storage. Like `da.store` but instead of
    returning a list of `dask.Delayed`, this function returns a list of `dask.Array`,
    which allows saving a subset of the data by slicing these arrays.
    """
    result = []

    if isinstance(sources, dask.array.core.Array):
        sources = [sources]
        targets = [targets]

    if len(sources) != len(targets):
        raise ValueError(
            "Different number of sources [%d] and targets [%d]"
            % (len(sources), len(targets))
        )

    if isinstance(regions, Sequence) or regions is None:
        regions = [regions]

    if len(sources) > 1 and len(regions) == 1:
        regions *= len(sources)

    if len(sources) != len(regions):
        raise ValueError(
            f"Different number of sources [{len(sources)}] and targets [{len(targets)}] than regions [{len(regions)}]"
        )

    for source, target, region in zip(sources, targets, regions):
        result.append(write_blocks(source, target, region))

    return result


def ensure_minimum_chunksize(array, chunksize):
    old_chunks = np.array(array.chunksize)
    new_chunks = old_chunks.copy()
    chunk_fitness = np.less(old_chunks, chunksize)
    if np.any(chunk_fitness):
        new_chunks[chunk_fitness] = np.array(chunksize)[chunk_fitness]
    return array.rechunk(new_chunks.tolist())
