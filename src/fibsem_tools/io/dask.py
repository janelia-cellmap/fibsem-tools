from os import PathLike
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import backoff
import dask
import dask.array as da
from dask.bag import from_sequence
import distributed
import numpy as np

from aiohttp import ServerDisconnectedError
from dask.array.core import (
    slices_from_chunks,
    normalize_chunks as normalize_chunks_dask,
)
from dask.array.optimization import fuse_slice
from dask.base import tokenize
from dask.core import flatten
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import fuse
from dask.utils import parse_bytes
from zarr.util import normalize_chunks as normalize_chunksize
from numpy.typing import NDArray, DTypeLike
import random
from fibsem_tools.io.core import access, read
from fibsem_tools.io.zarr import are_chunks_aligned
from dask import delayed

random.seed(0)


def fuse_delayed(tasks: dask.delayed) -> dask.delayed:
    """
    Apply task fusion optimization to tasks. Useful (or even required) because
    dask.delayed optimization doesn't do this step.
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
    Load slabs of an array into local memory, then create a dask array and rechunk that
    dask array, then store into chunked array storage.
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
def store_chunk(
    target: NDArray[Any], key: Tuple[slice, ...], value: NDArray[Any]
) -> Literal[0]:
    """
    A function inserted in a Dask graph for storing a chunk.

    Parameters
    ----------
    target: NDArray
        Where to store the value.
    key: Tuple[slice, ...]
        The location in the array for the value.
    value: NDArray
        The value to be stored.

    Examples
    --------
    """
    target[key] = value
    return 0


@backoff.on_exception(backoff.expo, (ServerDisconnectedError, OSError))
def store_value(
    target: NDArray[Any], key: Tuple[slice, ...], value: NDArray[Any]
) -> Literal[0]:
    """
    A function inserted in a Dask graph for storing a chunk.

    Parameters
    ----------
    target: NDArray
        Where to store the value.
    key: Tuple[slice, ...]
        The location in the array for the value.
    value: NDArray
        The value to be stored.

    Examples
    --------
    """
    target[key] = value
    return key


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

    # handle xarray
    if hasattr(source, "data") and isinstance(source.data, da.Array):
        source = source.data

    slices = slices_from_chunks(source.chunks)
    if region:
        slices = [fuse_slice(region, slc) for slc in slices]

    source_name = "store-source-" + tokenize(source)
    store_name = "store-" + tokenize(source)

    layers = {source_name: source.__dask_graph__()}
    deps = {source_name: set()}

    dsk = {}
    chunks = tuple((1,) * s for s in source.blocks.shape)

    for slce, key in zip(slices, flatten(source.__dask_keys__())):
        dsk[(store_name,) + key[1:]] = (
            ndwrapper,
            store_chunk,
            source.ndim,
            target,
            slce,
            key,
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
            f"""
            Different number of sources [{len(sources)}] and targets [{len(targets)}] 
            than regions [{len(regions)}]
            """
        )

    for source, target, region in zip(sources, targets, regions):
        result.append(write_blocks(source, target, region))

    return result


def write_blocks_delayed(
    source, target, region: Optional[Tuple[slice, ...]] = None
) -> Sequence[Any]:
    """
    Return a collection fo task each task returns the result of writing
    each chunk of `source` to `target`.
    """

    # handle xarray
    if hasattr(source, "data") and isinstance(source.data, da.Array):
        source = source.data

    slices = slices_from_chunks(source.chunks)
    if region:
        slices = [fuse_slice(region, slc) for slc in slices]
    blocks_flat = source.blocks.ravel()
    assert len(slices) == len(blocks_flat)
    return [
        delayed(store_value)(target, slce, block)
        for slce, block in zip(slices, blocks_flat)
    ]


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
    Scale a chunk size by an integer factor along each axis as much as possible without
    producing a chunk greater than a given size limit. Scaling will be applied to axes
    in decreasing order of length.

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
            f"""
            Could parse {item_size}, it should be type int or str, got 
            {type(item_size)}"""
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


@backoff.on_exception(backoff.expo, (ServerDisconnectedError, OSError))
def setitem(source, dest, sl):
    dest[sl] = source[sl]


def copy_from_slices(slices, source_array, dest_array):
    for sl in slices:
        setitem(source_array, dest_array, sl)


def copy_array(
    source: Union[PathLike, NDArray],
    dest: Union[PathLike, NDArray],
    chunk_size: Union[str, Tuple[int, ...]] = "100 MB",
    write_empty_chunks: bool = False,
    npartitions: int = 10000,
    randomize: bool = True,
    keep_attrs: bool = True,
):
    """
    Use Dask to copy data from one chunked array to another.

    Parameters
    ----------

    source: string, Pathlib.Path, array-like
        The source of the data to be copied. If this argument is a path or string,
        it is assumed to be url pointing to a resource that can be accessed via
        `fibsem_tools.io.core.read`. Otherwise, it is assumed to be a chunked
        array-like.

    dest: string, Pathlib.Path, array-like.
        The destination for the data to be copied. If this argument is a path or string,
        it is assumed to be url pointing to a resource that can be accessed via
        `fibsem_tools.io.core.access`. Otherwise, it is assumed to be a chunked
        array-like that supports writing / appending.

    chunk_size: tuple of ints or str
        The chunk size used for reading from the source data. If a string is given,
        it is assumed that this is a target size in bytes, and a chunk size will be
        chosen automatically to not exceed this size.

    write_empty_chunks: bool, defaults to False
        Whether empty chunks should be written to storage. Defaults to False.

    npartitions: int, defaults to 1000
        The array copying routine is wrapped in a dask bag. The npartitions parameter
        sets the the number of partitions of tha dask bag, and thus the degree of
        parallelism.

    randomize: bool, defaults to True
        If this parameter is True, then the dest array will be written in random
        order, which could minimize timeouts from cloud storage. This is untested and
        possibly superstitious.

    write_empty_chunks: bool, defaults to False
        Whether empty chunks of the source data will be written to dest.

    keep_attrs: bool, defaults to True
        Whether to copy the attributes of the source into dest.

    Returns
    -------

    A dask bag which, when computed, will copy data from source to dest.

    """
    if isinstance(source, PathLike):
        source_arr = read(source)
    else:
        source_arr = source

    if isinstance(dest, PathLike):
        dest_arr = access(dest, mode="a", write_empty_chunks=write_empty_chunks)
    else:
        dest_arr = dest

    # this should probably also be lazy.
    if keep_attrs:
        dest_arr.attrs.update(**source_arr.attrs)

    # assume we are given a size in bytes
    if isinstance(chunk_size, str):
        chunk_size_limit_bytes = parse_bytes(chunk_size)

        chunk_size = autoscale_chunk_shape(
            chunk_shape=dest_arr.chunks,
            array_shape=dest_arr.shape,
            size_limit=chunk_size_limit_bytes,
            dtype=dest_arr.dtype,
        )

    assert source_arr.shape == dest_arr.shape
    assert source_arr.dtype == dest_arr.dtype
    assert are_chunks_aligned(chunk_size, dest_arr.chunks)

    chunks_normalized = normalize_chunks_dask(chunk_size, shape=dest_arr.shape)
    slices = slices_from_chunks(chunks_normalized)

    # randomization to ensure that we don't create prefix hotspots when writing to
    # object storage
    if randomize:
        slices = random.sample(slices, len(slices))
    slice_bag = from_sequence(slices, npartitions=min(npartitions, len(slices)))

    return slice_bag.map_partitions(copy_from_slices, source_arr, dest_arr)


def pad_arrays(arrays, constant_values, stack=True):
    """
    Pad arrays with variable axis sizes. A bounding box is calculated across all the
    arrays and each sub-array is padded to fit within the bounding box. This is a light
    wrapper around dask.array.pad. If `stack` is True, the arrays will be combined into
    a larger array via da.stack.

    Parameters
    ----------
    arrays : iterable of dask arrays

    constant_values : any
        A number which specifies the fill value / mode to use when padding.

    stack: boolean
        Determines whether the result is a single dask array (stack=True) or a list of
        dask arrays (stack=False).

    Returns
    -------
    padded arrays and a list of paddings.
    """

    shapes = np.array([a.shape for a in arrays])
    bounds = shapes.max(0)
    pad_extent = [
        list(zip([0] * shapes.shape[1], (bounds - np.array(a.shape)).tolist()))
        for a in arrays
    ]

    # pad elements of the first axis differently
    def padfun(array, pad_width, constant_values):
        return np.stack(
            [
                np.pad(a, pad_width, constant_values=cv)
                for a, cv in zip(array, constant_values)
            ]
        )

    # If all the shapes are identical no padding is needed.
    if np.unique(shapes, axis=0).shape[0] == 1:
        padded = arrays
    else:
        padded = [
            a.map_blocks(
                padfun,
                pad_width=pad_extent[ind][1:],
                constant_values=constant_values,
                chunks=tuple(
                    c + p[1] - p[0] for c, p in zip(a.chunksize, pad_extent[ind])
                ),
                dtype=a.dtype,
            )
            for ind, a in enumerate(arrays)
        ]

    return padded, pad_extent
