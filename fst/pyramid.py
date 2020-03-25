import numpy as np
import dask.array as da
import dask
from collections import namedtuple
from xarray import DataArray
from typing import List, Union, Sequence, Callable


def even_padding(length: int, window: int) -> int:
    """
    Compute how much to add to `length` such that the resulting value is evenly divisible by `window`.
    
    Parameters
    ----------
    length : int
    window: int
    """
    return (window - (length % window)) % window


def logn(x: float, n: float) -> float:
    """
    Compute the logarithm of x base n.

    Parameters
    ----------
    x : float or int.
    n: float or int.

    Returns np.log(x) / np.log(n)
    -------

    """
    return np.log(x) / np.log(n)


def prepad(array: Union[np.array, da.array], scale_factors: Sequence, mode: str='reflect') -> da.array:
    """
    Pad an array such that its new dimensions are evenly divisible by some integer.

    Parameters
    ----------
    array: An ndarray that will be padded.

    scale_factors: An iterable of integers. The output array is guaranteed to have dimensions that are each evenly divisible
    by the corresponding scale factor, and chunks that are smaller than or equal to the scale factor (if the array has chunks)

    mode: String. The edge mode used by the padding routine. See `dask.array.pad` for more documentation.

    Returns a dask array with padded dimensions.
    -------

    """
    pw = tuple((0, even_padding(ax, scale))
               for ax, scale in zip(array.shape, scale_factors))

    result = None
    mode = 'reflect'
    if isinstance(array, dask.array.core.Array):
        result = da.pad(array, pw, mode=mode)
    else:
        result = da.pad(da.from_array(array), pw, mode=mode)

    # rechunk so that small extra chunks added by padding are fused into larger chunks    
    result = result.rechunk(result.chunksize)
    return result


def downscale(array: Union[np.array, da.array], reduction: Callable, scale_factors: Sequence) -> Union[np.array, da.array]:
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
    padded = prepad(array, scale_factors)
    return coarsen(reduction, padded, {d: s for d, s in enumerate(scale_factors)})


def get_downscale_depth(array: Union[np.array, da.array], scale_factors: Sequence) -> int:
    """
    For an array and a sequence of scale factors, calculate the maximum possible number of downscaling operations.
    """
    depths = {}
    for ax, s in enumerate(scale_factors):
        if s > 1:
            depths[ax] = np.ceil(logn(array.shape[ax], s)).astype('int')
    return min(depths.values())


def lazy_pyramid(array: Union[np.array, da.array], 
                 reduction: Callable, 
                 scale_factors: Sequence, 
                 preserve_dtype: bool=True, 
                 max_depth: int=5) -> List[DataArray]:
    """
    Lazily generate an image pyramid

    Parameters
    ----------
    array: ndarray to be downscaled.

    reduction: a function that aggregates data over windows.

    scale_factors: an iterable of integers that specifies how much to downscale each axis of the array.

    preserve_dtype: Boolean, determines whether lower levels of the pyramid are coerced to the same dtype as the input.

    Returns a list of DataArrays, one per level of downscaling. These DataArrays have `coords` properties that track the changing offset (if any)
    induced by the downsampling operation. Additionally, the scale factors are stored each DataArray's attrs propery under the key `scale_factors` 
    -------

    """
    assert len(scale_factors) == array.ndim
    scale = (1,) * array.ndim
    
    if hasattr(array, 'coords'):
        base_coords = tuple(map(np.array, array.coords.values()))
        base_attrs = array.attrs
        dims = array.dims
    else:
        base_coords=tuple(offset + np.arange(dim, dtype='float32')
                                for dim, offset in zip(array.shape, get_downsampled_offset(scale)))
        dims = None

    result = [DataArray(data=da.asarray(array),
                        coords=base_coords,
                        attrs={'scale_factors': scale, **base_attrs},
                        dims=dims)]

    # figure out the maximum depth
    levels = range(1, get_downscale_depth(array, scale_factors))[:max_depth]

    for l in levels:
        scale = tuple(s ** l for s in scale_factors)
        arr = downscale(array, reduction, scale)
        if preserve_dtype:
            arr = arr.astype(array.dtype)
        new_coords = tuple(offset + bc[:dim] * sc
                                for dim, bc, offset, sc in zip(arr.shape, base_coords, get_downsampled_offset(scale), scale))
                                
        result.append(DataArray(data=arr,
                                coords = new_coords,
                                attrs={'scale_factors': scale, **base_attrs}, 
                                dims=dims))
    return result


def get_downsampled_offset(scale_factors: Sequence) -> np.array:
    """
    For a given number of dimensions and a sequence of downscale factors, calculate the starting offset of the downscaled
    array in the units of the full-resolution data.
    """
    ndim = len(scale_factors)
    return np.mgrid[tuple(slice(scale_factors[dim]) for dim in range(ndim))].mean(tuple(range(1, 1 + ndim)))