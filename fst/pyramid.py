import numpy as np
import dask.array as da
import dask
from collections import namedtuple

def even_padding(length, window):
    """
    Compute how much to add to `length` such that the resulting value is evenly divisible by `window`.
    """
    return (window - (length % window)) % window


def logn(x, n):
    """
    Compute the logarithm of x base n.

    Parameters
    ----------
    x : numeric value.
    n: numeric value.

    Returns np.log(x) / np.log(n)
    -------

    """
    return np.log(x) / np.log(n)


def prepad(array, scale_factors, mode='reflect'):
    """
    Pad an array such that its new dimensions are evenly divisible by some integer.

    Parameters
    ----------
    array: An ndarray that will be padded.

    scale_factors: An iterable of integers. The output array is guaranteed to have dimensions that each evenly divisible
    by the corresponding scale factor.

    mode: String. The edge mode used by the padding routine. See `dask.array.pad` for more documentation.

    Returns a dask array with padded dimensions.
    -------

    """
    pw = tuple((0, even_padding(ax, scale)) for ax, scale in zip(array.shape, scale_factors))

    result = None
    mode = 'reflect'
    if isinstance(array, dask.array.core.Array):
        result = da.pad(array, pw, mode=mode)
    else:
        result = da.pad(da.from_array(array), pw, mode=mode)

    # rechunk so that padding does not change number of chunks
    new_chunks = list(result.chunks)
    for ind, c in enumerate(result.chunks):
        if pw[ind][-1] > 0:
            tmp = list(c)
            tmp[-2] += tmp[-1]
            new_chunks[ind] = tuple(tmp[:-1])

    new_chunks = tuple(new_chunks)
    result = result.rechunk(new_chunks)
    return result


def downscale(array, reduction, scale_factors):
    """
    Downscale an array using windowed aggregation. This function is a light wrapper for `dask.array.coarsen`.

    Parameters
    ----------
    array: The narray to be downscaled.
    reduction: The function to apply to each window of the array.
    scale_factors: A list if ints specifying how much to downscale the array per dimension.

    Returns the downscaled version of the input as a dask array.
    -------

    """
    from dask.array import coarsen
    return coarsen(reduction, prepad(array, scale_factors), {d: s for d, s in enumerate(scale_factors)})


def get_downscale_depth(array, scale_factors):
    """
    For an array and a sequence of scale factors, calculate the maximum possible number of downscaling operations.
    """
    depths = {}
    for ax, s in enumerate(scale_factors):
        if s > 1:
            depths[ax] = np.ceil(logn(array.shape[ax], s)).astype('int')
    return min(depths.values())


def lazy_pyramid(array, reduction, scale_factors, preserve_dtype=True):
    """
    Lazily generate an image pyramid

    Parameters
    ----------
    array: ndarray to be downscaled.

    reduction: a function that aggregates data over windows.

    scale_factors: an iterable of integers that specifies how much to downscale each axis of the array.

    preserve_dtype: Boolean, determines whether lower levels of the pyramid are coerced to the same dtype as the input.

    Returns a list of named tuples, one per level of downscaling, each with an `array` field (containing a dask array
    representing a downscaled image), a `scaling_factors` field containing the scaling factors used to create that
    array, and an `offset` field which contains the offset into the first pixel of the downsampled image.
    -------

    """
    assert len(scale_factors) == array.ndim

    # level 0 is the original
    Level = namedtuple('level', 'array scale_factors offset')
    result = [Level(array=da.asarray(array), scale_factors=(1,) * array.ndim, offset=(0,) * array.ndim)]

    # figure out the maximum depth
    levels = range(1, get_downscale_depth(array, scale_factors))
    for l in levels:
        scale = tuple(s ** l for s in scale_factors)
        arr = downscale(array, reduction, scale)
        if preserve_dtype:
            arr = arr.astype(array.dtype)
        result.append(Level(array=arr,
                            scale_factors=scale,
                            offset=get_downsampled_offset(scale)))
    return result


def get_downsampled_offset(scale_factors):
    """
    For a given number of dimension and a sequence of downscale factors, calculate the starting offset of the downscaled
    array in the units of the full-resolution data.
    """
    ndim = len(scale_factors)
    return np.mgrid[tuple(slice(scale_factors[dim]) for dim in range(ndim))].mean(tuple(range(1, 1 + ndim)))
