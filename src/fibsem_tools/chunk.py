from typing import Literal, Sequence, Tuple

import numpy as np
from dask.utils import parse_bytes
from xarray import DataArray
from zarr.util import normalize_chunks as normalize_chunksize


def chunk_grid_shape(
    array_shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Get the shape of the chunk grid of a Zarr array.
    """
    return tuple(np.ceil(np.divide(array_shape, chunk_shape)).astype("int").tolist())


def are_chunks_aligned(
    source_chunks: Tuple[int, ...], dest_chunks: Tuple[int, ...]
) -> bool:
    assert len(source_chunks) == len(dest_chunks)
    return all(
        s_chunk % d_chunk == 0 for s_chunk, d_chunk in zip(source_chunks, dest_chunks)
    )


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


def resolve_slice(slce: slice, interval: Tuple[int, int]) -> slice:
    """
    Given a `slice` object and a half-open interval indexed by the slice,
    return a `slice` object with `start`, `stop` and `step` attributes that are all integers.
    """
    step = 1 if slce.step is None else slce.step
    sliced_interval = tuple(range(*interval))[slce]
    return slice(sliced_interval[0], sliced_interval[-1] + 1, step)


def resolve_slices(
    slces: Sequence[slice], intervals: Sequence[Tuple[int, int]]
) -> Tuple[slice, ...]:
    """
    Convenience function for applying `resolve_slice` to a collection of `slice` objects and a collection of half-open intervals.
    """
    return tuple(map(lambda v: resolve_slice(*v), zip(slces, intervals)))


def interval_remainder(
    interval_a: Tuple[int, int], interval_b: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Repeat interval_b until it forms an interval that is larger than or equal to interval_a.
    Return the number of elements that must be added to the start and end of interval_a to match this length.

    If interval_b is an integer multiple of interval_a, return (0,0)
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
    arrays: Sequence[DataArray],
    chunks: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...], Literal["auto"]],
) -> Tuple[Tuple[int, ...], ...]:
    """
    Normalize a chunk specification, given a list of arrays.

    Parameters
    ----------

    arrays: Sequence[DataArray]
        The list of arrays to define chunks for.
    chunks: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...], Literal["auto"]]
        The specification of chunks. This parameter is either a tuple of tuple of ints,
        in which case it is already normalized and it passes right through, or it is
        a tuple of ints, which will be "broadcast" to the length of `arrays`, or it is
        the string "auto", in which case the existing chunks on the arrays with be used
        if they are chunked, and otherwise chunks will be set to the shape of each
        array.

    Returns
    -------
        Tuple[Tuple[int, ...], ...]
    """
    result: Tuple[Tuple[int, ...]] = ()
    if chunks == "auto":
        for arr in arrays:
            if arr.chunks is None:
                result += (arr.shape,)
            else:
                # use the chunksize property of the underlying dask array
                result += (arr.data.chunksize,)
    elif all(isinstance(c, tuple) for c in chunks):
        result = chunks
    else:
        all_ints = all((isinstance(c, int) for c in chunks))
        if all_ints:
            result = (chunks,) * len(arrays)
        else:
            msg = f"All values in chunks must be ints. Got {chunks}"
            raise ValueError(msg)

    assert len(result) == len(arrays)
    assert tuple(map(len, result)) == tuple(
        x.ndim for x in arrays
    ), "Number of chunks per array does not equal rank of arrays"
    return result
