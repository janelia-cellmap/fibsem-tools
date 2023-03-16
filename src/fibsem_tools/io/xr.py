import xarray as xr
import numpy as np
import numpy.typing as npt
from typing import Tuple


def stt_coord(length: int, dim: str, scale: float, translate: float, unit: str):
    """
    Create a coordinate variable parametrized by a shape, a scale, a translation, and
    a unit. The translation is applied after the scaling.
    """
    return xr.DataArray(
        (np.arange(length) * scale) + translate, dims=(dim,), attrs={"units": unit}
    )


def stt_from_array(
    data: npt.ArrayLike,
    dims: Tuple[str, ...],
    scales: Tuple[float, ...],
    translates: Tuple[float, ...],
    units: Tuple[str, ...],
) -> xr.DataArray:

    """
    Create a DataArray with coordinates parametrized by a shape, a sequence of dims,
    a sequence of scales, a sequence of translates, and a sequence of units from an
    input array.
    """
    coords = []
    for idx, s in enumerate(data.shape):
        coords.append(stt_coord(s, dims[idx], scales[idx], translates[idx], units[idx]))

    return xr.DataArray(data, dims=dims, coords=coords)
