import numpy as np
import dask.array as da
from typing import Iterable, Union


def padstack(arrays: Iterable[da.Array], mode: str = 'minimum', constant_values: Union[int, float] = 0) -> da.Array:
    """
    Stack arrays with variable axis sizes. A bounding box is calculated across all the arrays and each sub-array is
    padded to fit within the bounding box. This is a light wrapper around dask.array.pad.

    Parameters
    ----------
    arrays : An iterable collection of dask arrays.
    mode: The fill mode to use. See `dask.array.pad`
    constant_values : The value to fill when padding images, if necessary.

    Returns a dask array containing the entire dataset represented by the individual arrays in `arrays`.
    -------

    """

    shapes = np.array([a.shape for a in arrays])
    bounds = shapes.max(0)
    pad_extent = [
        list(zip((0,) * shapes.shape[1], bounds - np.array(a.shape))) for a in arrays
    ]
    if mode == 'constant':
        padded = [(da.pad(a, pad_width=pad_extent[ind], mode=mode, constant_values=constant_values)) for ind, a in enumerate(arrays)]
    else:
        padded = [(da.pad(a, pad_width=pad_extent[ind], mode=mode)) for ind, a in enumerate(arrays)]

    stacked = da.stack(padded)
    return stacked
