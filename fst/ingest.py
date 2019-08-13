import numpy as np
import dask.array as da
from typing import Iterable, Union


def padstack(arrays: Iterable[da.Array], fill_value: Union[int, float] = 0) -> da.Array:
    """
    Stack arrays with variable axis sizes. A bounding box is calculated across all the arrays

    Parameters
    ----------
    arrays : An iterable collection of dask arrays.
    fill_value : The value to fill when padding images, if necessary.

    Returns a dask array containing the entire dataset represented by the individual arrays in `arrays`.
    -------

    """
    shapes = np.array([a.shape for a in arrays])
    bounds = shapes.max(0)
    pad_extent = [
        list(zip((0,) * shapes.shape[1], bounds - np.array(a.shape))) for a in arrays
    ]
    padded = [
        da.pad(a, pad_extent[ind], mode="constant", constant_values=fill_value)
        for ind, a in enumerate(arrays)
    ]
    stacked = da.stack(padded)
    return stacked
