from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal, Sequence

from datatree import DataTree
import zarr
from xarray import DataArray, Dataset
from zarr.errors import ReadOnlyError

from fibsem_tools.types import PathLike
from fibsem_tools.io.n5.hierarchy import cosem, neuroglancer
from ..zarr import access as access_zarr, get_url
import dask.array as da
from dask.base import tokenize


def to_dask(
    arr: zarr.Array,
    chunks: Literal["auto"] | Sequence[int] = "auto",
    inline_array: bool = True,
    **kwargs: Any,
) -> da.Array:
    """
    Create a Dask array from a Zarr array. This is a very thin wrapper around `dask.array.from_array`

    Parameters
    ----------

    arr : zarr.Array

    chunks : Literal['auto'] | Sequence[int]
        The chunks to use for the output Dask array. It may be tempting to use the chunk size of the
        input array for this parameter, but be advised that Dask performance suffers when arrays
        have too many chunks, and Zarr arrays routinely have too many chunks by this definition.

    inline_array : bool, default is `True`
        Whether the Zarr array should be inlined in the Dask compute graph. See documentation for
        `dask.array.from_array` for details.

    **kwargs : Any
        Additional keyword arguments for `dask.array.from_array`

    Returns
    -------

    dask.array.Array
    """
    if kwargs.get("name") is None:
        kwargs["name"] = f"{get_url(arr)}-{tokenize(arr)}"
    darr = da.from_array(arr, chunks=chunks, inline_array=inline_array, **kwargs)
    return darr


class N5FSStorePatched(zarr.N5FSStore):
    """
    Patch delitems to delete without checking if to-be-deleted keys exist.
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


def create_dataarray(
    element: zarr.Array,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataArray:
    """
    Create a DataArray from a Zarr array (wrapping an N5 dataset). Coordinates will be inferred from
    the array attributes.
    """
    array_attrs = element.attrs.asdict()
    # cosem first, then neuroglancer
    if coords == "auto":
        if "transform" in array_attrs:
            return cosem.create_dataarray(
                array=element, use_dask=use_dask, chunks=chunks
            )
        else:
            return neuroglancer.create_dataarray(
                array=element, use_dask=use_dask, chunks=chunks
            )
    else:
        if use_dask:
            wrapped = to_dask(element, chunks=chunks)
        else:
            wrapped = element
        return DataArray(wrapped, coords=coords, attrs=attrs, name=name)


def create_datatree(
    element: zarr.Group,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataTree:
    if coords != "auto":
        msg = (
            "This function does not support values of `coords` other than `auto`. "
            f"Got {coords}. This may change in the future."
        )
        raise NotImplementedError(msg)

    if name is None:
        name = element.basename

    nodes: dict[str, Dataset | DataArray | DataTree | None] = {
        name: create_dataarray(
            array,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=None,
            name="data",
        )
        for name, array in element.arrays()
    }
    if attrs is None:
        root_attrs = element.attrs.asdict()
    else:
        root_attrs = attrs
    # insert root element
    nodes["/"] = Dataset(attrs=root_attrs)
    dtree = DataTree.from_dict(nodes, name=name)
    return dtree


def is_n5(array: zarr.core.Array) -> bool:
    """
    Check if a Zarr array is backed by N5 storage.
    """
    return isinstance(array.store, (zarr.N5Store, zarr.N5FSStore))


def to_xarray(
    element: Any,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    coords: Any = "auto",
    name: str | None = None,
) -> DataArray | DataTree:
    if isinstance(element, zarr.Group):
        return create_datatree(
            element,
            chunks=chunks,
            coords=coords,
            attrs=attrs,
            use_dask=use_dask,
            name=name,
        )
    elif isinstance(element, zarr.Array):
        return create_dataarray(
            element,
            chunks=chunks,
            coords=coords,
            attrs=attrs,
            use_dask=use_dask,
            name=name,
        )
    else:
        raise ValueError(
            "This function only accepts instances of zarr.Group and zarr.Array. ",
            f"Got {type(element)} instead.",
        )
