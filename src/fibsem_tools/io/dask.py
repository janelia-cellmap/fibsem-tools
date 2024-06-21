from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Literal

    import distributed
    import zarr

import random
from fibsem_tools.type import PathLike

import backoff
import dask
import dask.array as da
import numpy as np
from aiohttp import ServerDisconnectedError
from dask import delayed
from dask.bag import Bag
from dask.array.core import (
    normalize_chunks as normalize_chunks_dask,
)
from dask.array.core import (
    slices_from_chunks,
)
from dask.array.optimization import fuse_slice
from dask.bag import from_sequence
from dask.base import tokenize
from dask.core import flatten
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import fuse
from dask.utils import parse_bytes

from fibsem_tools.chunk import are_chunks_aligned, autoscale_chunk_shape, resolve_slices
from fibsem_tools.io.core import access, read

random.seed(0)


def fuse_delayed(tasks: dask.delayed) -> dask.delayed:
    """
    Apply task fusion optimization to tasks. Useful (or even required) because
    dask.delayed optimization doesn't do this step.
    """
    dsk_fused, deps = fuse(dask.utils.ensure_dict(tasks.dask))
    return Delayed(tasks._key, dsk_fused)


def sequential_rechunk(
    source: Any,
    target: Any,
    slab_size: tuple[int],
    intermediate_chunks: tuple[int],
    client: distributed.Client,
    num_workers: int,
) -> list[None]:
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
    target: zarr.Array, key: tuple[slice, ...], value: np.ndarray
) -> Literal[0]:
    """
    A function inserted in a Dask graph for storing a chunk.

    Parameters
    ----------
    target: NDArray
        Where to store the value.
    key: tuple[slice, ...]
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
    target: zarr.Array, key: tuple[slice, ...], value: np.ndarray
) -> Literal[0]:
    """
    A function inserted in a Dask graph for storing a chunk.

    Parameters
    ----------
    target: NDArray
        Where to store the value.
    key: tuple[slice, ...]
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


def write_blocks(source, target, region: tuple[slice, ...] | None) -> da.Array:
    """
    Return a dask array with where each chunk contains the result of writing
    each chunk of `source` to `target`.
    """

    # handle xarray DataArrays
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


def store_blocks(sources, targets, regions: slice | None = None) -> list[da.Array]:
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
        msg = (
            f"Number of sources ({len(sources)}) does not match the "
            f"number of regions  ({len(regions)})"
        )
        raise ValueError(msg)

    for source, target, region in zip(sources, targets, regions):
        result.append(write_blocks(source, target, region))

    return result


def write_blocks_delayed(
    source, target, region: tuple[slice, ...] | None = None
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
    if len(slices) != len(blocks_flat):
        msg = "Number of slices does not match the number of blocks"
        raise ValueError(msg)
    return [
        delayed(store_value)(target, slce, block)
        for slce, block in zip(slices, blocks_flat)
    ]


@backoff.on_exception(backoff.expo, (ServerDisconnectedError, OSError))
def setitem(
    source: da.Array,
    dest: zarr.Array,
    selection: tuple[slice, ...],
    *,
    chunk_safe: bool = True,
):
    if chunk_safe and hasattr(dest, "chunks"):
        selection_resolved = resolve_slices(
            selection, tuple((0, s) for s in dest.shape)
        )

        for sel, cnk, shape in zip(selection_resolved, dest.chunks, dest.shape):
            if sel.start % cnk != 0 or ((sel.stop != shape) and sel.stop % cnk != 0):
                msg = (
                    f"Planned writes are not chunk-aligned. Destination array has chunks sized {dest.chunks} "
                    f"but the requested selection {selection} partially crosses chunk boundaries. "
                    "Either call this function with `chunk_safe=False`, or align your writes to the "
                    "chunk boundaries of the destination."
                )
                raise ValueError(msg)
    dest[selection] = source[selection]


def copy_from_slices(slices, source_array, dest_array):
    for sl in slices:
        setitem(source_array, dest_array, sl)


def copy_array(
    source: PathLike | (np.ndarray[Any, Any] | zarr.Array),
    dest: PathLike | (np.ndarray[Any, Any] | zarr.Array),
    *,
    chunk_size: str | tuple[int, ...] = "100 MB",
    write_empty_chunks: bool = False,
    npartitions: int = 10000,
    randomize: bool = True,
    keep_attrs: bool = True,
) -> Bag:
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
    source_arr = read(source) if isinstance(source, PathLike) else source

    dest_arr = (
        access(dest, mode="a", write_empty_chunks=write_empty_chunks)
        if isinstance(dest, (str, Path))
        else dest
    )

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

    if source_arr.shape != dest_arr.shape:
        msg = "Shapes are not equal"
        raise ValueError(msg)
    if source_arr.dtype != dest_arr.dtype:
        msg = "Datatypes are not equal"
        raise ValueError(msg)
    if not are_chunks_aligned(chunk_size, dest_arr.chunks):
        msg = "Chunks are not aligned"
        raise ValueError(msg)

    chunks_normalized = normalize_chunks_dask(chunk_size, shape=dest_arr.shape)
    slices = slices_from_chunks(chunks_normalized)

    # randomization to ensure that we don't create prefix hotspots when writing to
    # object storage
    if randomize:
        slices = random.sample(slices, len(slices))
    slice_bag = from_sequence(slices, npartitions=min(npartitions, len(slices)))

    return slice_bag.map_partitions(copy_from_slices, source_arr, dest_arr)


def pad_arrays(arrays, constant_values):
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
    def padfun(
        array: np.ndarray[Any, Any],
        pad_width: tuple[tuple[int, int], ...],
        constant_values: tuple[Any, ...],
    ) -> np.ndarray[Any.Any]:
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
