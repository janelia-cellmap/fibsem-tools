from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    import zarr
    from pydantic_ome_ngff.v04.multiscale import Group
    from xarray import DataArray

from xarray_ome_ngff.array_wrap import DaskArrayWrapper, ZarrArrayWrapper
from xarray_ome_ngff.v04.multiscale import model_group, read_array

"""
def model_group(
    *,
    arrays: dict[str, DataArray],
    transform_precision: int | None = None,
    chunks: tuple[int, ...] | tuple[tuple[int, ...]] | Literal["auto"] = "auto",
    compressor: Codec | None = Zstd(3),
    fill_value: Any = 0,
) -> MultiscaleGroup:
"""


def multiscale_group(
    arrays: dict[str, DataArray],
    *,
    chunks: tuple[tuple[int, ...]] | Literal["auto"] = "auto",
    **kwargs,
) -> Group:
    """
    Create a model of an OME-NGFF v04 multiscale group from a dict of DataArrays.
    A very thin wrapper around `xarray_ome_ngff.v04.multiscale.model_group`
    """
    return model_group(arrays=arrays, chunks=chunks, **kwargs)


def create_dataarray(
    array: zarr.Array,
    use_dask: bool = True,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
    name: str | None = None,
) -> DataArray:
    wrapper = DaskArrayWrapper(chunks=chunks) if use_dask else ZarrArrayWrapper()

    result = read_array(array=array, array_wrapper=wrapper)
    # read_array doesn't take the name kwarg at the moment
    if name is not None:
        result.name = name
    return result
