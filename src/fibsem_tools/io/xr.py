import xarray
import numpy as np
import numpy.typing as npt
from typing import Any, Tuple, Sequence


def flip(data: xarray.DataArray, dims: Sequence[str] = []) -> xarray.DataArray:
    """
    Reverse the data backing a DataArray along the specified dimension(s).
    """
    flip_selector = ()
    for dim in data.dims:
        if dim in dims:
            flip_selector += (slice(None, None, -1),)
        else:
            flip_selector += (slice(None),)
    return data.copy(data=data[flip_selector].data)


def stt_coord(length: int, dim: str, scale: float, translate: float, unit: str):
    """
    Create a coordinate variable parametrized by a shape, a scale, a translation, and
    a unit. The translation is applied after the scaling.
    """
    return xarray.DataArray(
        (np.arange(length) * scale) + translate, dims=(dim,), attrs={"units": unit}
    )


def stt_from_array(
    data: npt.ArrayLike,
    dims: Tuple[str, ...],
    scales: Tuple[float, ...],
    translates: Tuple[float, ...],
    units: Tuple[str, ...],
    **kwargs: Any,
) -> xarray.DataArray:
    """
    Create a DataArray with coordinates parametrized by a shape, a sequence of dims,
    a sequence of scales, a sequence of translations, and a sequence of units from an
    input array.
    """
    coords = []
    for idx, s in enumerate(data.shape):
        coords.append(stt_coord(s, dims[idx], scales[idx], translates[idx], units[idx]))

    return xarray.DataArray(data, dims=dims, coords=coords, **kwargs)
