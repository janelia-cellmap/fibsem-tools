import numpy as np
import dask.array as da
from typing import Iterable, Union
from numbers import Number
from dask import delayed

def padstack(arrays: Iterable[da.Array], constant_values: Union[int, float, str] = 0) -> da.Array:
    """
    Stack arrays with variable axis sizes. A bounding box is calculated across all the arrays and each sub-array is
    padded to fit within the bounding box. This is a light wrapper around dask.array.pad.

    Parameters
    ----------
    arrays : An iterable collection of dask arrays.
        constant_values : The value to fill when padding images

    Returns a dask array containing the entire dataset represented by the individual arrays in `arrays`.
    -------

    """

    shapes = np.array([a.shape for a in arrays])
    bounds = shapes.max(0)
    pad_extent = [
        list(zip((0,) * shapes.shape[1], bounds - np.array(a.shape))) for a in arrays
    ]

    if isinstance(constant_values, Number):
        padded = [(da.pad(a, pad_width=pad_extent[ind], mode='constant', constant_values=constant_values)) for ind, a in enumerate(arrays)]
    elif constant_values == 'minimum-minus-one':
        delmin = delayed(np.min)
        # this will be bad if the minimum value in the data hits the floor of the datatype
        fill_value = int(min(delayed(delmin(a) for a in arrays).compute()) - 1)

        padded = [(da.pad(a, pad_width=pad_extent[ind], mode='constant', constant_values=fill_value)) for ind, a in enumerate(arrays)]

    stacked = da.stack(padded)
    return stacked
