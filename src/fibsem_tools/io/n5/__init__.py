from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal, Sequence

import zarr
from xarray import DataArray
from zarr.errors import ReadOnlyError

from fibsem_tools.types import PathLike

from ..zarr import access as access_zarr

N5_AXES_3D = ["x", "y", "z"]


class N5FSStorePatched(zarr.N5FSStore):
    """
    Patch delitems to delete "blindly", i.e. without checking if to-be-deleted keys exist.
    This is temporary and should be removed when
    https://github.com/zarr-developers/zarr-python/issues/1336 is resolved.
    """

    def delitems(self, keys: Sequence[str]) -> None:
        if self.mode == "r":
            raise ReadOnlyError()
        try:  # should be much faster
            nkeys = [self._normalize_key(key) for key in keys]
            # rm errors if you pass an empty collection
            self.map.delitems(nkeys)
        except FileNotFoundError:
            nkeys = [self._normalize_key(key) for key in keys if key in self]
            # rm errors if you pass an empty collection
            if len(nkeys) > 0:
                self.map.delitems(nkeys)


DEFAULT_N5_STORE = N5FSStorePatched


def access(store: PathLike, path: PathLike, **kwargs: Any) -> zarr.Group | zarr.Array:
    store = DEFAULT_N5_STORE(store, **kwargs.get("storage_options", {}))
    return access_zarr(store, path, **kwargs)


def is_n5(array: zarr.core.Array) -> bool:
    """
    Check if a Zarr array is backed by N5 storage.
    """
    return isinstance(array.store, (zarr.N5Store, zarr.N5FSStore))


from fibsem_tools.io.n5.hierarchy import cosem, neuroglancer


def create_dataarray(
    array: zarr.Array,
    use_dask: bool = True,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
) -> DataArray:
    """
    Create a DataArray from a Zarr array (wrapping an N5 dataset). Coordinates will be inferred from
    the array attributes.
    """
    array_attrs = array.attrs.asdict()
    # cosem first, then neuroglancer
    if "transform" in array_attrs:
        return cosem.create_dataarray(array=array, use_dask=use_dask, chunks=chunks)
    else:
        return neuroglancer.create_dataarray(
            array=array, use_dask=use_dask, chunks=chunks
        )
