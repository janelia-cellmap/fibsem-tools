from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

import zarr
from xarray import DataArray
from xarray_ome_ngff.array_wrap import DaskArrayWrapper, ZarrArrayWrapper
from xarray_ome_ngff.v04.multiscale import read_array


def create_dataarray(
    array: zarr.Array,
    use_dask: bool = True,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
) -> DataArray:
    if use_dask:
        wrapper = DaskArrayWrapper(chunks=chunks)
    else:
        wrapper = ZarrArrayWrapper()

    return read_array(array=array, array_wrapper=wrapper)
