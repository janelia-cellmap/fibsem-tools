from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from fibsem_tools.type import PathLike

import dask.array as da
import zarr
from dask.base import tokenize
from xarray import DataArray, Dataset, DataTree
from zarr.errors import ReadOnlyError

from fibsem_tools.io.n5.hierarchy import cosem, neuroglancer
from fibsem_tools.io.zarr.core import access as access_zarr
from fibsem_tools.io.zarr.core import get_url


def to_dask(
    arr: zarr.Array,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    inline_array: bool = True,
    **kwargs: Any,
) -> da.Array:
    """
    Create a Dask array from a Zarr array. This is a very thin wrapper around `dask.array.from_array`

    Parameters
    ----------

    arr : zarr.Array

    chunks : Literal['auto'] | tuple[int, ...]
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
    return da.from_array(arr, chunks=chunks, inline_array=inline_array, **kwargs)


class N5FSStorePatched(zarr.N5FSStore):
    """
    A version of `zarr.N5FSStore` with a more efficient implementation of `delitems`.
    The `delitems` routine in `zarr.N5FSStore` checks if objects exist before deleting them.
    The implementation of `delitems` used here does not check if objects exist before
    attempting to delete them, which is better for performance.

    This situation is temporary and should be removed when
    https://github.com/zarr-developers/zarr-python/issues/1336 is resolved.
    """

    def delitems(self, keys: Sequence[str]) -> None:
        if self.mode == "r":
            raise ReadOnlyError
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
    """
    Access N5 groups or datasets (arrays) via `zarr-python`.

    Parameters
    ----------
    store: str | Path
        The root of the hierarchy
    path: str | path
        The relative path to the element to open.
    **kwargs: Any
        Additional keyword arguments passed to `access_zarr`, other than any keyword arguments
        using the `storage_options` key, which is passed to the constructor for the n5 store.
    """
    store = DEFAULT_N5_STORE(store, **kwargs.pop("storage_options", {}))
    return access_zarr(store, path, **kwargs)


def create_dataarray(
    element: zarr.Array,
    *,
    use_dask: bool = True,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
    coords: Any = "auto",
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataArray:
    """
    Create a DataArray from a Zarr array (wrapping an N5 dataset). Coordinates will be inferred from
    the array attributes.

    Parameters
    ----------
    element: zarr.Array
        The zarr.Array object that will be wrapped by `xarray.DataArray`
    *
    use_dask: bool = True
        Whether to wrap `element` in a dask array before passing it to `xarray.DataArray`.
    chunks: tuple[int, ...] | Literal["auto"] = "auto"
        The chunks for the array, if `use_dask` is `True`
    coords: Any = "auto"
        `DataArray`-compatible coordinates for `element`. This will override any coordinate
        inference.
    attrs: dict[str, Any] | None = None
        Attributes for the `DataArray`. If `None` (the default), then the `attrs` attribute of
        `element` will be used.
    name: str | None = None
        The name for the `DataArray`.
    """
    array_attrs = element.attrs.asdict()
    # cosem first, then neuroglancer
    if coords == "auto":
        if "transform" in array_attrs:
            result = cosem.create_dataarray(
                array=element, use_dask=use_dask, chunks=chunks, name=name
            )

        else:
            result = neuroglancer.create_dataarray(
                array=element, use_dask=use_dask, chunks=chunks, name=name
            )
        if attrs is not None:
            result.attrs = attrs
    else:
        wrapped = to_dask(element, chunks=chunks) if use_dask else element
        result = DataArray(wrapped, coords=coords, attrs=attrs, name=name)

    return result


def create_datatree(
    element: zarr.Group,
    *,
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
        ).to_dataset()
        for name, array in element.arrays()
    }
    root_attrs = element.attrs.asdict() if attrs is None else attrs
    # insert root element
    nodes["/"] = Dataset(attrs=root_attrs)
    return DataTree.from_dict(nodes, name=name)


def is_n5(array: zarr.Array) -> bool:
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

    return create_dataarray(
        element,
        chunks=chunks,
        coords=coords,
        attrs=attrs,
        use_dask=use_dask,
        name=name,
    )
