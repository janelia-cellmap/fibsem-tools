from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union, Literal

from xarray import DataArray
from .transform import STTransform
from fibsem_tools.io.util import normalize_chunks
from cellmap_schemas.multiscale.neuroglancer_n5 import Group


def multiscale_group(
    *,
    arrays: dict[str, DataArray],
    chunks: Union[tuple[tuple[int, ...]], Literal["auto"]],
    **kwargs,
) -> Group:
    """
    Create a model of a Neuroglancer-compatible N5 multiscale group from a collection of
    DataArrays

    Parameters
    ----------

    arrays: dict[str, DataArray]
        The data to model.
    chunks: The chunks for each Zarr array in the group.


    """
    _chunks = normalize_chunks(arrays.values(), chunks)

    transforms = tuple(STTransform.from_xarray(array) for array in arrays.values())

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
