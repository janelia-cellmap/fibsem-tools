from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    import zarr

import warnings

import dask.array as da
import numpy as np
from cellmap_schemas.multiscale.neuroglancer_n5 import Group
from xarray import DataArray

from fibsem_tools.chunk import normalize_chunks
from fibsem_tools.coordinate import stt_coord, stt_from_array
from fibsem_tools.io.zarr.core import access_parent

N5_AXES_3D = ["x", "y", "z"]


def model_group(
    arrays: dict[str, DataArray],
    *,
    chunks: tuple[tuple[int, ...], ...] | tuple[int, ...] | Literal["auto"] = "auto",
    **kwargs,
) -> Group:
    """
    Create a model of a Neuroglancer-compatible N5 multiscale group from a collection of
    DataArrays

    Parameters
    ----------

    arrays: dict[str, DataArray]
        The data to model.
    chunks: tuple[tuple[int, ...], ...] | tuple[int, ...] | Literal["auto"] = "auto",
        The chunks for each Zarr array in the group.
    **kwargs:
        Additional keyword arguments passed to `Group.from_arrays`
    """
    _chunks = normalize_chunks(arrays.values(), chunks)

    transforms = tuple(stt_from_array(array) for array in arrays.values())
    base_transform = transforms[0]
    nonzero_translate = any(v != 0 for v in base_transform.translate)

    if nonzero_translate:
        msg = (
            f"Non-zero translation parameters ({base_transform.translate}) detected in the base "
            "coordinates. Be advised that this translation parameter will not be stored, due to limitations "
            "of the metadata format you are using."
        )
        warnings.warn(msg)

    return Group.from_arrays(
        arrays=tuple(arrays.values()),
        paths=tuple(arrays.keys()),
        scales=[t.scale for t in transforms],
        axes=transforms[0].axes,
        units=transforms[0].units,
        dimension_order=transforms[0].order,
        chunks=_chunks[0],
        **kwargs,
    )


def create_dataarray(
    array: zarr.Array,
    *,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    use_dask: bool = True,
    name: str | None = None,
) -> DataArray:
    """
    Create a DataArray from an N5 dataset that uses Neuroglancer-compatible N5 metadata.

    Parameters
    ----------

    array: zarr.Array
        A handle to the Zarr array
    use_dask: bool = True
        Whether to wrap the result in a dask array. Default is True.
    chunks: Literal["auto"] | tuple[int, ...] = "auto"
        The chunks to use for the returned array. When `use_dask` is `False`, then `chunks` must be
        "auto".
    name: str | None
        The name for the resulting array.
    """
    group_model = Group.from_zarr(access_parent(array))
    array_model = group_model.members[array.basename]
    members_sorted_by_shape = dict(
        sorted(
            group_model.members.items(), key=lambda v: np.prod(v[1].shape), reverse=True
        )
    )
    scale_index = tuple(members_sorted_by_shape.keys()).index(array.basename)
    base_scales = group_model.attributes.pixelResolution.dimensions
    array_scale_factor = group_model.attributes.scales[scale_index]
    translation = tuple(
        np.arange(array_scale_factor[0]).mean() * scale for scale in base_scales
    )
    if use_dask:
        array_wrapped = da.from_array(array, chunks=chunks)
    else:
        if chunks != "auto":
            msg = f"If use_dask is False, then chunks must be 'auto'. Got {chunks} instead."
            raise ValueError(msg)
        array_wrapped = array

    pixelResolution = array_model.attributes.pixelResolution
    dims_in = N5_AXES_3D
    dims_out = dims_in[::-1]

    transes = dict(zip(dims_in, translation))
    scales = dict(zip(dims_in, pixelResolution.dimensions))
    units = {ax: pixelResolution.unit for ax in dims_in}

    coords = tuple(
        stt_coord(array.shape[idx], ax, scales[ax], transes[ax], units[ax])
        for idx, ax in enumerate(dims_out)
    )
    return DataArray(
        array_wrapped,
        dims=dims_out,
        coords=coords,
        attrs=array.attrs.asdict(),
        name=name,
    )
