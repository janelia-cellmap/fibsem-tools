from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal
    from dask.array.core import Array as DArray
    import numpy.typing as npt
    from xarray import DataArray

import numpy as np
from dask.utils import parse_bytes
from zarr.util import guess_chunks
from zarr.util import normalize_chunks as normalize_chunksize


def chunk_grid_shape(
    array_shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Get the shape of the chunk grid of a regularly chunked array.
    """
    return tuple(np.ceil(np.divide(array_shape, chunk_shape)).astype("int").tolist())


def are_chunks_aligned(
    source_chunks: tuple[int, ...], dest_chunks: tuple[int, ...]
) -> bool:
    if len(source_chunks) != len(dest_chunks):
        msg = "Length of source chunks does not match length of dest chunks."
        raise ValueError(msg)
    return all(
        s_chunk % d_chunk == 0 for s_chunk, d_chunk in zip(source_chunks, dest_chunks)
    )


def ensure_minimum_chunksize(array: DArray, chunksize: tuple[int, ...]) -> DArray:
    old_chunks = np.array(array.chunksize)
    new_chunks = old_chunks.copy()
    chunk_fitness = np.less(old_chunks, chunksize)
    if np.any(chunk_fitness):
        new_chunks[chunk_fitness] = np.array(chunksize)[chunk_fitness]
    return array.rechunk(new_chunks.tolist())


def autoscale_chunk_shape(
    chunk_shape: tuple[int, ...],
    array_shape: tuple[int, ...],
    size_limit: str | int,
    dtype: npt.DTypeLike,
) -> tuple[int, ...]:
    """
    Scale a chunk size by an integer factor along each axis as much as possible without
    producing a chunk greater than a given size limit. Scaling will be applied to axes
    in decreasing order of length.

    Parameters
    ----------
    chunk_shape : tuple[int, ...]
        The base chunk shape to scale. The resulting chunk shape will be factorizable by this
        value.
    array_shape : tuple[int, ...]
        The shape of the array to create chunks for.
    size_limit : str | int
        The maximum size, in bytes, for a single chunk.
    dtype : np.DtypeLike
        The datatype of the elements in the array. Used for calculating how large chunks are
        in memory.


    Returns
    -------
    tuple[int, ...]
        The original chunk size after each element has been multiplied by some integer.

    """

    item_size = np.dtype(dtype).itemsize

    if isinstance(size_limit, str):
        size_limit_bytes = parse_bytes(size_limit)
    elif isinstance(size_limit, int):
        size_limit_bytes = size_limit
    else:
        msg = f"Could not parse {item_size}. Expected type int or str, got {type(item_size)}"
        raise TypeError(msg)

    if size_limit_bytes < 1:
        msg = f"Chunk size limit {size_limit} is too small."
        raise ValueError(msg)

    normalized_chunk_shape = normalize_chunksize(chunk_shape, array_shape, item_size)
    result = normalized_chunk_shape
    chunk_size_bytes = np.prod(normalized_chunk_shape) * item_size

    size_ratio = size_limit_bytes / chunk_size_bytes
    chunk_grid_shape = np.ceil(np.divide(array_shape, chunk_shape)).astype("int")

    if size_ratio < 1:
        return result

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

    return tuple(np.multiply(scale_vector, normalized_chunk_shape).tolist())


def resolve_slice(slce: slice, interval: tuple[int, int]) -> slice:
    """
    Given a `slice` object and a half-open interval indexed by the slice,
    return a `slice` object with `start`, `stop` and `step` attributes that are all integers.
    """
    step = 1 if slce.step is None else slce.step
    sliced_interval = tuple(range(*interval))[slce]
    return slice(sliced_interval[0], sliced_interval[-1] + 1, step)


def resolve_slices(
    slces: Sequence[slice], intervals: Sequence[tuple[int, int]]
) -> tuple[slice, ...]:
    """
    Convenience function for applying `resolve_slice` to a collection of `slice` objects and a collection of half-open intervals.
    """
    return tuple(resolve_slice(*v) for v in zip(slces, intervals))


def interval_remainder(
    interval_a: tuple[int, int], interval_b: tuple[int, int]
) -> tuple[int, int]:
    """
    Repeat `interval_b` until it forms an interval that is larger than or equal to `interval_a`.
    Return the number of elements that must be added to the start and end of `interval_a` to match
    this length. If `interval_b` is an integer multiple of `interval_a`, return (0,0)
    """
    start_a, stop_a = interval_a

    start_b, stop_b = interval_b
    len_b = stop_b - start_b

    start_diff = start_b - start_a
    stop_diff = stop_a - stop_b

    if start_diff <= 0:
        start_scaled = start_b
    else:
        scale_bottom = np.ceil(start_diff / len_b).astype(int)
        start_scaled = start_b - len_b * scale_bottom

    if stop_diff <= 0:
        stop_scaled = stop_b
    else:
        scale_top = np.ceil(stop_diff / len_b).astype("int")
        stop_scaled = stop_b + len_b * scale_top

    return start_a - start_scaled, stop_scaled - stop_a


def normalize_chunks(
    arrays: Iterable[DataArray],
    chunks: tuple[tuple[int, ...], ...] | tuple[int, ...] | Literal["auto"],
) -> tuple[tuple[int, ...], ...]:
    """
    Normalize a chunk specification, given an iterable of DataArrays.

    Parameters
    ----------

    arrays: Sequence[DataArray]
        The list of arrays to define chunks for. They should be sorted by shape,
        in descending order. I.e., the first array should be the largest.
    chunks: Literal["auto"] | tuple[int, ...] | tuple[tuple[int, ...], ...]
        The specification of chunks. This parameter is either a tuple of tuple of ints,
        in which case it is already normalized and it passes right through, or it is
        a tuple of ints, which will be "broadcast" to the length of `arrays`, or it is
        the string "auto", in which case the existing chunks on the arrays with be used
        if they are chunked, and otherwise chunks will be set to a value based on the size and
        data type of the largest array.

    Returns
    -------
        tuple[tuple[int, ...], ...]
    """
    result: tuple[tuple[int, ...], ...] = ()
    arrays_tuple = tuple(arrays)
    if chunks == "auto":
        # duck typing check for all dask arrays
        if all(hasattr(a.data, "chunksize") for a in arrays_tuple):
            # use the chunksize for each array
            result = tuple(a.data.chunksize for a in arrays_tuple)
        else:
            # guess chunks for the largest array, and use that for all the others
            largest = sorted(
                arrays_tuple, key=lambda v: np.prod(v.shape), reverse=True
            )[0]
            result = (
                guess_chunks(largest.shape, typesize=largest.dtype.itemsize),
            ) * len(arrays_tuple)

    elif all(isinstance(c, tuple) for c in chunks):
        chunks = cast(tuple[tuple[int, ...], ...], chunks)
        if all(all(isinstance(sub, int) for sub in c) for c in chunks):
            result = chunks
        else:
            msg = f"Not all inner elements of chunks were integers: {chunks}"
            raise ValueError(msg)
    else:
        all_ints = all(isinstance(c, int) for c in chunks)
        if all_ints:
            result = cast(tuple[tuple[int, ...], ...], (chunks,) * len(arrays_tuple))
        else:
            msg = f"All values in chunks must be ints. Got {chunks}"
            raise ValueError(msg)

    if len(result) != len(arrays_tuple):
        msg = "Length of arrays does not match the length of chunks."
        raise ValueError(msg)
    if tuple(map(len, result)) != tuple(x.ndim for x in arrays_tuple):
        msg = "Number of chunks per array does not equal rank of arrays"
        raise ValueError(msg)
    return result
