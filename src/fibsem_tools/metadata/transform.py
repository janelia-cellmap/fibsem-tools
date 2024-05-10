from __future__ import annotations

from typing import Literal, Sequence

from cellmap_schemas.multiscale.cosem import STTransform
from xarray import DataArray

import fibsem_tools.io.xr as fsxr


def stt_to_coords(transform: STTransform, shape: tuple[int, ...]) -> tuple[DataArray]:
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
    if transform.order == "C":
        axes = transform.axes
    else:
        axes = reversed(transform.axes)
    return tuple(
        fsxr.stt_coord(
            shape[idx],
            dim=k,
            scale=transform.scale[idx],
            translate=transform.translate[idx],
            unit=transform.units[idx],
        )
        for idx, k in enumerate(axes)
    )


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
            raise ValueError(
                f"The coordinate with dims = {c.dims} does not have enough elements to calculate "
                "a scaling transformation. A minimum of 2 elements are needed."
            )
        axes.append(str(c.dims[0]))
        # default unit is m
        units.append(c.attrs.get("units", "m"))
        translate.append(float(c[0]))
        scale.append(abs(float(c[1]) - float(c[0])))
        assert scale[-1] > 0

    return STTransform(
        axes=axes, units=units, translate=translate, scale=scale, order=order
    )


def stt_from_array(array: DataArray, reverse_axes: bool = False) -> STTransform:
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
    output_order = "C"
    if reverse_axes:
        orderer = slice(-1, None, -1)
        output_order = "F"

    return stt_from_coords(tuple(array.coords.values())[orderer], output_order)
