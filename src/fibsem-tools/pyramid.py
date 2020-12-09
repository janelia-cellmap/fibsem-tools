import numpy as np
import dask.array as da
from xarray import DataArray
from typing import Any, List, Optional, Union, Sequence, Callable
from scipy.interpolate import interp1d
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.utils import SerializableLock

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


def prepad(
    array: Union[np.array, da.array],
    scale_factors: Sequence,
    mode: str = "reflect",
    rechunk: bool = True,
) -> da.array:
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
    pw = tuple(
        (0, even_padding(ax, scale)) for ax, scale in zip(array.shape, scale_factors)
    )

    result = da.pad(array, pw, mode=mode)

    # rechunk so that small extra chunks added by padding are fused into larger chunks, but only if we had to add chunks after padding
    if rechunk and np.any(pw):
        new_chunks = tuple(
            np.multiply(
                scale_factors, np.ceil(np.divide(result.chunksize, scale_factors))
            ).astype("int")
        )
        result = result.rechunk(new_chunks)
    if hasattr(array, "coords"):
        new_coords = {}
        for p, k in zip(pw, array.coords):
            old_coord = array.coords[k]
            if np.diff(p) == 0:
                new_coords[k] = old_coord
            else:
                extended_coords = interp1d(
                    np.arange(len(old_coord.values)),
                    old_coord.values,
                    fill_value="extrapolate",
                )(np.arange(len(old_coord.values) + p[-1])).astype(old_coord.dtype)
                new_coords[k] = DataArray(
                    extended_coords, dims=k, attrs=old_coord.attrs
                )
        result = DataArray(
            result, coords=new_coords, dims=array.dims, attrs=array.attrs
        )
    return result


def downscale(
    array: Union[np.array, da.array],
    reduction: Callable,
    scale_factors: Sequence[int],
    **kwargs
) -> DataArray:
    """
    Downscale an array using windowed aggregation. This function is a light wrapper for `dask.array.coarsen`.

    Parameters
    ----------
    array: The narray to be downscaled.
    
    reduction: The function to apply to each window of the array.
    
    scale_factors: A list if ints specifying how much to downscale the array per dimension.    

    **kwargs: extra kwargs passed to dask.array.coarsen

    Returns the downscaled version of the input as a dask array.
    -------

    """
    from dask.array import coarsen

    padded = prepad(array, scale_factors)
    return coarsen(
        reduction, padded, {d: s for d, s in enumerate(scale_factors)}, **kwargs
    )


def get_downscale_depth(
    array: Union[np.array, da.array], scale_factors: Sequence[int]
) -> int:
    """
    For an array and a sequence of scale factors, calculate the maximum possible number of downscaling operations.
    """
    depths = {}
    for ax, s in enumerate(scale_factors):
        if s > 1:
            depths[ax] = np.ceil(logn(array.shape[ax], s)).astype("int")
    return min(depths.values())


def lazy_pyramid(
    array: Union[np.array, da.array],
    reduction: Callable = np.mean,
    scale_factors: Union[Sequence[int], int] = 2,
    preserve_dtype: bool = True,
    max_depth: int = 5,
) -> List[DataArray]:
    """
    Lazily generate an image pyramid

    Parameters
    ----------
    array: ndarray to be downscaled.

    reduction: a function that aggregates data over windows.

    scale_factors: an iterable of integers that specifies how much to downscale each axis of the array. 

    preserve_dtype: boolean, defaults to True, determines whether lower levels of the pyramid are coerced to the same dtype as the input. This assumes that
    the reduction function accepts a "dtype" kwarg, e.g. numpy.mean(x, dtype='int').

    max_depth: int, The number of downscaling operations to perform.

    Returns a list of DataArrays, one per level of downscaling. These DataArrays have `coords` properties that track the changing offset (if any)
    induced by the downsampling operation. Additionally, the scale factors are stored each DataArray's attrs propery under the key `scale_factors` 
    -------

    """
    if isinstance(scale_factors, int):
        scale_factors = (scale_factors,) * array.ndim
    else:
        assert len(scale_factors) == array.ndim
    scale = (1,) * array.ndim

    # figure out the maximum depth
    levels = range(1, 1 + get_downscale_depth(array, scale_factors))[:max_depth]

    if hasattr(array, "coords"):
        # if the input is a xarray.DataArray, assign a new variable to the DataArray and use the variable
        # `array` to refer to the data property of that array
        data = array.data
        dims = array.dims
        # ensure that key order matches dimension order
        base_coords = {d: array.coords[d] for d in dims}
        base_attrs = array.attrs
    else:
        data = array
        dims = [str(x) for x in range(data.ndim)]
        base_coords = {
            dim: DataArray(offset + np.arange(s, dtype="float32"), dims=dim)
            for dim, s, offset in zip(dims, array.shape, get_downsampled_offset(scale))
        }
        base_attrs = {}

    result = [
        DataArray(
            data=da.asarray(data),
            coords=base_coords,
            dims=dims,
            attrs={"scale_factors": scale, **base_attrs},
        )
    ]

    for l in levels:
        scale = tuple(s ** l for s in scale_factors)

        if preserve_dtype:
            arr = downscale(data, reduction, scale).astype(array.dtype)
        else:
            arr = downscale(data, reduction, scale)

        # hideous
        new_coords = tuple(
            DataArray(
                (offset * (base_coords[bc][1] - base_coords[bc][0]))
                + base_coords[bc][:s] * sc,
                name=base_coords[bc].name,
                attrs=base_coords[bc].attrs,
            )
            for s, bc, offset, sc in zip(
                arr.shape, base_coords, get_downsampled_offset(scale), scale
            )
        )

        result.append(
            DataArray(
                data=arr,
                coords=new_coords,
                attrs={"scale_factors": scale, **base_attrs},
            )
        )
    return result


def get_downsampled_offset(scale_factors: Sequence) -> np.array:
    """
    For a given number of dimensions and a sequence of downscale factors, calculate the starting offset of the downscaled
    array in the units of the full-resolution data.
    """
    ndim = len(scale_factors)
    return np.mgrid[tuple(slice(scale_factors[dim]) for dim in range(ndim))].mean(
        tuple(range(1, 1 + ndim))
    )


def downscale_slice(sl: slice, scale: int) -> slice:
    """
    Downscale the start, stop, and step of a slice by an integer factor. Ceiling division is used, i.e.
    downscale_slice(Slice(0, 10, None), 3) returns Slice(0, 4, None).
    """

    start, stop, step = sl.start, sl.stop, sl.step
    if start:
        start = int(np.ceil(sl.start / scale))
    if stop:
        stop = int(np.ceil(sl.stop / scale))
    if step:
        step = int(np.ceil(sl.step / scale))
    result = slice(start, stop, step)

    return result


def slice_span(sl: slice) -> int:
    """
    Measure the length of a slice
    """
    return sl.stop - sl.start


def blocked_pyramid(
    arr, block_size: Sequence, scale_factors: Sequence = (2, 2, 2), **kwargs
):
    full_pyr = lazy_pyramid(arr, scale_factors=scale_factors, **kwargs)
    slices = slices_from_chunks(normalize_chunks(block_size, arr.shape))
    absolute_block_size = tuple(map(slice_span, slices[0]))

    results = []
    for idx, sl in enumerate(slices):
        regions = [
            tuple(map(downscale_slice, sl, tuple(np.power(scale_factors, exp))))
            for exp in range(len(full_pyr))
        ]
        if tuple(map(slice_span, sl)) == absolute_block_size:
            pyr = lazy_pyramid(arr[sl], scale_factors=scale_factors, **kwargs)
        else:
            pyr = [full_pyr[l][r] for l, r in enumerate(regions)]
        assert len(pyr) == len(regions)
        results.append((regions, pyr))
    return results


def blocked_store(sources, targets, chunks=None):
    stores = []
    for slices, source in sources:
        if chunks is not None:
            rechunked_sources = [
                s.data.rechunk(chunks) for s, z in zip(source, targets)
            ]
        elif hasattr(targets[0], "chunks"):
            rechunked_sources = [
                s.data.rechunk(z.chunks) for s, z in zip(source, targets)
            ]
        else:
            rechunked_sources = [s.data for s in source]

        stores.append(
            da.store(
                rechunked_sources,
                targets,
                lock=SerializableLock(),
                regions=slices,
                compute=False,
            )
        )
    return stores


def subsample_reduce(a, axis=None):
    """
    Coarsening by subsampling, compatible with da.coarsen
    """
    if axis is None:
        return a
    else:
        samples = []
        for ind,s in enumerate(a.shape):
            if ind in axis:
                samples.append(slice(0, 1, None))
            else:
                samples.append(slice(None))        

        return a[tuple(samples)].squeeze()


def mode_reduce(a: Any, axis: Optional[int]=None) -> Any:
    """
    Coarsening by computing the n-dimensional mode, compatible with da.coarsen. If input is all 0s, the mode is not computed.
    """
    from scipy.stats import mode
    if axis is None:
         return a
    elif a.max() == 0:
        return np.min(a, axis)
    else:
        transposed = a.transpose(*range(0, a.ndim, 2), *range(1, a.ndim, 2))
        reshaped = transposed.reshape(*transposed.shape[:a.ndim//2], -1)
        modes = mode(reshaped, axis=reshaped.ndim-1).mode
        result = modes.squeeze(axis=-1)
        # sometimes we get a chunk with a single element along an axis, but we want to keep it
        # squeeze would remove that axis by default, which screws up the dimensionality of the result
        # this check ensures that we pad the dimensions after over-zealous squeezing
        # if result.ndim < a.ndim // 2:
        #    result = np.expand_dims(result, result.ndim)
        return result

