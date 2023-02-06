from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import backoff
import dask
import dask.array as da
import distributed
import numpy as np
from aiohttp import ServerDisconnectedError
from dask.array.core import slices_from_chunks
from dask.array.optimization import fuse_slice
from dask.base import tokenize
from dask.core import flatten
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import fuse
from dask.utils import is_arraylike, parse_bytes
from zarr.util import normalize_chunks as normalize_chunksize
from numpy.typing import NDArray, DTypeLike


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


def autoscale_chunk_shape(
    chunk_shape: Tuple[int, ...],
    array_shape: Tuple[int, ...],
    size_limit: Union[str, int],
    dtype: DTypeLike,
):

    """
    Scale a chunk size by an integer factor along each axis as much as possible without producing a chunk greater than a
    given size limit. Scaling will be applied to axes in decreasing order of length.

    Parameters
    ----------

    chunk_shape : type
        description
    array_shape : type
        description
    size_limit : type
        description
    dtype : type
        description


    Returns
    -------

    tuple of ints
        The original chunk size after each element has been multiplied by some integer.


    Examples
    --------

    """

    item_size = np.dtype(dtype).itemsize

    if isinstance(size_limit, str):
        size_limit_bytes = parse_bytes(size_limit)
    elif isinstance(size_limit, int):
        size_limit_bytes = size_limit
    else:
        raise TypeError(
            f"Could parse {item_size}, it should be type int or str, got {type(item_size)}"
        )

    if size_limit_bytes < 1:
        raise ValueError(f"Chunk size limit {size_limit} is too small.")

    normalized_chunk_shape = normalize_chunksize(chunk_shape, array_shape, item_size)
    result = normalized_chunk_shape
    chunk_size_bytes = np.prod(normalized_chunk_shape) * item_size

    size_ratio = size_limit_bytes / chunk_size_bytes
    chunk_grid_shape = np.ceil(np.divide(array_shape, chunk_shape)).astype("int")

    if size_ratio < 1:
        return result
    else:
        target_nchunks = np.ceil(size_ratio).astype("int")

    # operate in chunk grid coordinates
    # start with 1 chunk
    scale_vector = [
        1,
    ] * len(chunk_shape)
    sorted_idx = reversed(np.argsort(chunk_grid_shape))
    # iterate over axes in order of length
    for idx in sorted_idx:
        # compute how many chunks are still needed
        chunks_needed = target_nchunks - np.prod(scale_vector)
        # compute number of chunks available along this axis
        chunks_available = np.prod(scale_vector) * chunk_grid_shape[idx]
        if chunks_needed > chunks_available:
            scale_vector[idx] = chunk_grid_shape[idx]
        else:
            scale_vector[idx] = max(
                1, np.floor_divide(chunks_needed, np.prod(scale_vector))
            )
            break

    result = tuple(np.multiply(scale_vector, normalized_chunk_shape).tolist())

    return result
