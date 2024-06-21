from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, Any

    import zarr
    from pydantic_ome_ngff.v04.multiscale import Group
    from xarray import DataArray

from xarray_ome_ngff.array_wrap import DaskArrayWrapper, ZarrArrayWrapper
import xarray_ome_ngff.v04.multiscale as multiscale


def model_group(
    arrays: dict[str, DataArray],
    *,
    chunks: tuple[tuple[int, ...]] | tuple[int, ...] | Literal["auto"] = "auto",
    **kwargs: Any,
) -> Group:
    """
    Create a model of an OME-NGFF v04 multiscale group from a dict of DataArrays.
    This is a very thin wrapper around `xarray_ome_ngff.v04.multiscale.model_group`

    Parameters
    ----------
    arrays: dict[str, DataArray]
        A dictionary of `DataArray` that represents a multiscale image.
    chunks: tuple[[tuple[int, ...]], ...] | tuple[int, ...] | Literal["auto"] = "auto"
        The chunks to use for the Zarr arrays. The default value of "auto" will choose a chunk size
        based on the size and dtype of the largest array, and use it for all the arrays.
    **kwargs:
        Additional keyword arguments passed to `model_group`.

    Returns
    -------
    Group
        A `GroupSpec` instance that models a multiscale group, and can be used to create
        a Zarr group in storage.
    """
    return multiscale.model_group(arrays=arrays, chunks=chunks, **kwargs)


def create_dataarray(
    array: zarr.Array,
    *,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
    use_dask: bool = True,
    name: str | None = None,
) -> DataArray:
    """
    Create a DataArray from a Zarr array OME-NGFF version 0.4 metadata.

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
    wrapper = DaskArrayWrapper(chunks=chunks) if use_dask else ZarrArrayWrapper()

    result = multiscale.read_array(array=array, array_wrapper=wrapper)
    # read_array doesn't take the name kwarg at the moment
    if name is not None:
        result.name = name
    return result
