import numpy as np
import dask.array as da
import dask


def even_padding(length, window):
    return (window - (length % window)) % window


def logn(x,n):
    return np.log(x) / np.log(n)


def prepad(image, scale_factors: tuple):
    pw = tuple((0,even_padding(ax, scale)) for ax, scale in zip(image.shape, scale_factors))
    result = None
    mode = 'reflect'
    if isinstance(image, dask.array.core.Array):
        result = da.pad(image, pw, mode=mode).rechunk((-1,) * im.ndim)
    else:
        result = da.pad(da.from_array(image), pw, mode=mode).rechunk((-1,) * image.ndim)
    return result


def downscale(image, reduction, scale_factors):
    from dask.array import coarsen
    return coarsen(reduction, prepad(image, scale_factors), {d:s for d,s in enumerate(scale_factors)})


def get_downscale_depth(image, scale_factors):
    depths = {}
    for ax, s in enumerate(scale_factors):
        if s > 1:
            depths[ax] = np.ceil(logn(image.shape[ax], s)).astype('int')
    return min(depths.values())


def lazy_pyramid(image, reduction, scale_factors):
    assert len(scale_factors) == image.ndim 
    # figure out the maximum depth
    result = [da.from_array(image)]
    levels = range(1, get_downscale_depth(image, scale_factors))
    for l in levels:
        scale = tuple(s ** l for s in scale_factors)
        result.append(downscale(image, reduction, scale))
    return result


def get_downsampled_offset(ndim, scale_factors):
    return np.mgrid[tuple(slice(scale_factors[dim]) for dim in range(ndim))].mean(tuple(range(1, 1 + ndim)))
    
