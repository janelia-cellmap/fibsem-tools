from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cellmap_schemas.multiscale.cosem import STTransform
from xarray import DataArray

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from fibsem_tools.type import ArrayLike


def stt_coord(
    length: int, dim: str, scale: float, translate: float, unit: str
) -> DataArray:
    """
    Create a coordinate variable parametrized by a shape, a scale, a translation, and
    a unit. The translation is applied after the scaling.
    """
    return DataArray(
        (np.arange(length) * scale) + translate, dims=(dim,), attrs={"units": unit}
    )


def stt_array(
    data: ArrayLike,
    dims: Sequence[str],
    scales: Sequence[float],
    translates: Sequence[float],
    units: Sequence[str],
    **kwargs: Any,
) -> DataArray:
    """
    Create a DataArray with coordinates parametrized by a shape, a sequence of dims,
    a sequence of scales, a sequence of translations, and a sequence of units from an
    input array.
    """
    coords = []
    for idx, s in enumerate(data.shape):
        coords.append(stt_coord(s, dims[idx], scales[idx], translates[idx], units[idx]))

    return DataArray(data, dims=dims, coords=coords, **kwargs)


def flip(data: DataArray, dims: Sequence[str] = []) -> DataArray:
    """
    Reverse the data backing a DataArray along the specified dimension(s).
    """
    flip_selector: tuple[slice, ...] = ()
    for dim in data.dims:
        if dim in dims:
            flip_selector += (slice(None, None, -1),)
        else:
            flip_selector += (slice(None),)
    return data.copy(data=data[flip_selector].data)


def stt_from_coords(
    coords: Sequence[DataArray], order: Literal["C", "F"] = "C"
) -> STTransform:
    """
    Generate a spatial transform from coordinates.

    Parameters
    ----------

    coords: Sequence[xarray.DataArray]
        A sequence of 1D DataArrays, one per dimension.
    order: Literal["C", "F"]
        The array indexing order to use. "C" denotes numpy-style lexicographic indexing
    Returns
    -------

    STTransform
        An instance of STTransform that is consistent with `coords`.
    """

    axes = []
    units = []
    scale = []
    translate = []

    for c in coords:
        if len(c) < 2:
            msg = (
                f"The coordinate with dims = {c.dims} does not have enough elements to calculate "
                "a scaling transformation. A minimum of 2 elements are needed."
            )
            raise ValueError(msg)
        axes.append(str(c.dims[0]))
        # default unit is m
        units.append(c.attrs.get("units", "m"))
        translate.append(float(c[0]))
        scale.append(abs(float(c[1]) - float(c[0])))
        if any(tuple(s <= 0 for s in scale)):
            msg = f"Invalid scale: {scale}. Scale must be greater than 0."
            raise ValueError(msg)

    return STTransform(
        axes=tuple(axes),
        units=tuple(units),
        translate=tuple(translate),
        scale=tuple(scale),
        order=order,
    )


def stt_from_array(array: DataArray, *, reverse_axes: bool = False) -> STTransform:
    """
    Generate a spatial transform from a DataArray.

    Parameters
    ----------

    array: xarray.DataArray
        A DataArray with coordinates that can be expressed as scaling + translation
        applied to a regular grid.
    reverse_axes: boolean, default=False
        If `True`, the order of the `axes` in the spatial transform will
        be reversed relative to the order of the dimensions of `array`, and the
        `order` field of the resulting STTransform will be set to "F". This is
        designed for compatibility with N5 tools.

    Returns
    -------

    STTransform
        An instance of STTransform that is consistent with the coordinates defined
        on the input DataArray.
    """

    orderer = slice(None)
    output_order: Literal["C", "F"] = "C"
    if reverse_axes:
        orderer = slice(-1, None, -1)
        output_order = "F"

    return stt_from_coords(tuple(array.coords.values())[orderer], output_order)


def stt_to_coords(
    transform: STTransform, shape: tuple[int, ...]
) -> tuple[DataArray, ...]:
    """
    Given an array shape, return a list of DataArrays representing a
    bounded coordinate grid derived from this transform. This list can be used as
    the `coords` argument to construct a DataArray.

    Parameters
    ----------
    transform: STTransform

    shape: Tuple[int, ...]
        The shape of the coordinate grid, e.g. the size of the array that will be
        annotated with coordinates.

    Returns
    -------
    tuple[DataArray]
        A tuple of DataArrays, one per axis.

    """
    axes = transform.axes if transform.order == "C" else reversed(transform.axes)
    return tuple(
        stt_coord(
            shape[idx],
            dim=k,
            scale=transform.scale[idx],
            translate=transform.translate[idx],
            unit=transform.units[idx],
        )
        for idx, k in enumerate(axes)
    )
