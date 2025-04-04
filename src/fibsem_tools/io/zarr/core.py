from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal

    from numcodecs.abc import Codec

    from fibsem_tools.type import PathLike

import os
from pathlib import Path

import dask.array as da
import zarr
from dask.base import tokenize
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from xarray import DataArray, Dataset, DataTree
from xarray_ome_ngff.core import get_parent
from zarr.errors import ReadOnlyError
from zarr.indexing import BasicIndexer
from zarr.storage import BaseStore, FSStore

from fibsem_tools.io.zarr.hierarchy import ome_ngff

noneslice = slice(None)


def parse_url(url: str) -> tuple[str, str]:
    """
    Parse a url with the format <prefix>.zarr/<postfix> into a (<prefix>.zarr, <postfix>) tuple.
    """
    suffix = ".zarr"
    sep = "/"
    parts = url.split(sep)
    suffixed = []
    for idx, p in enumerate(parts):
        if p.endswith(suffix):
            suffixed.append(idx)

    if len(suffixed) == 0:
        msg = f"None of the parts of the url {url} end with {suffix}."
        raise ValueError(msg)

    if len(suffixed) > 1:
        msg = f"Too many parts of the url {url} end with {suffix}. Expected just 1, but found {len(suffix)}."
        raise ValueError(msg)

    return sep.join(parts[: suffixed[0] + 1]), sep.join(parts[(suffixed[0] + 1) :])


class FSStorePatched(FSStore):
    """
    Patch delitems to delete without checking if to-be-deleted keys exist.
    This is temporary and should be removed when
    https://github.com/zarr-developers/zarr-python/issues/1336
    is resolved.
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


DEFAULT_ZARR_STORE = FSStorePatched


def same_compressor(arr: zarr.Array, compressor: Codec) -> bool:
    """
    Determine if the compressor associated with an array is the same as a different
    compressor.

    arr: A zarr array
    compressor: a Numcodecs compressor, e.g. GZip(-1)
    return: True or False, depending on whether the zarr array's compressor matches the
    parameters (name, level) of the compressor.
    """
    comp = arr.compressor.compressor_config
    return comp["id"] == compressor.codec_id and comp["level"] == compressor.level


noneslice = slice(None)


def same_array_props(
    arr: zarr.Array,
    shape: tuple[int, ...],
    dtype: str,
    compressor: Any,
    chunks: tuple[int, ...],
) -> bool:
    """

    Determine if a zarr array has properties that match the input properties.

    arr: A zarr array
    shape: A tuple. This will be compared with arr.shape.
    dtype: A numpy dtype. This will be compared with arr.dtype.
    compressor: A numcodecs compressor, e.g. GZip(-1). This will be compared with the
      compressor of arr.
    chunks: A tuple. This will be compared with arr.chunks
    return: True if all the properties of arr match the kwargs, False otherwise.
    """
    return (
        (arr.shape == shape)
        & (arr.dtype == dtype)
        & same_compressor(arr, compressor)
        & (arr.chunks == chunks)
    )


def array_from_dask(arr: da.Array) -> zarr.Array:
    """
    Return the zarr array that was used to create a dask array using
    `da.from_array(zarr_array)`
    """
    maybe_array: zarr.Array | tuple[Any, zarr.Array, tuple[slice, ...]] = next(
        iter(arr.dask.values())
    )
    if isinstance(maybe_array, zarr.Array):
        return maybe_array
    return maybe_array[1]


def get_url(node: zarr.Group | zarr.Array) -> str:
    store = node.store
    if hasattr(store, "path"):
        if hasattr(store, "fs"):
            protocol = (
                store.fs.protocol[0]
                if isinstance(store.fs.protocol, Sequence)
                else store.fs.protocol
            )
        else:
            protocol = "file"

        # fsstore keeps the protocol in the path, but not s3store
        store_path = store.path.split("://")[-1] if "://" in store.path else store.path
        return f"{protocol}://{os.path.join(store_path, node.path)}"
    else:
        msg = (
            f"The store associated with this object has type {type(store)}, which "
            "cannot be resolved to a url"
        )
        raise ValueError(msg)


def get_store(path: PathLike) -> BaseStore:
    if isinstance(path, Path):
        path = str(path)

    return DEFAULT_ZARR_STORE(path)


def access(
    store: PathLike | BaseStore, path: PathLike, **kwargs: Any
) -> zarr.Array | zarr.Group:
    if isinstance(store, (Path, str)):
        store = get_store(store)

    # set default dimension separator to /
    if "shape" in kwargs and "dimension_separator" not in kwargs:
        kwargs["dimension_separator"] = "/"

    if isinstance(path, Path):
        path = str(path)

    attrs = kwargs.pop("attrs", {})
    access_mode = kwargs.pop("mode", "a")

    array_or_group = zarr.open(store, path=path, **kwargs, mode=access_mode)

    if access_mode != "r" and len(attrs) > 0:
        array_or_group.attrs.update(attrs)
    return array_or_group


def chunk_keys(
    array: zarr.Array, region: slice | tuple[slice, ...] = noneslice
) -> Generator[str, None, None]:
    """
    Get the keys for all the chunks in a Zarr array as a generator of strings.

    Parameters
    ----------
    array: zarr.core.Array
        The zarr array to get the chunk keys from
    region: slice | tuple[slice, ...]
        The region in the zarr array get chunks keys from. Defaults to `slice(None)`, which
        will result in all the chunk keys being returned.
    Returns
    -------
    Generator[str, None, None]

    """
    indexer = BasicIndexer(region, array)
    chunk_coords = (idx.chunk_coords for idx in indexer)
    return (array._chunk_key(cc) for cc in chunk_coords)


def to_dask(
    arr: zarr.Array,
    *,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
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
    return da.from_array(arr, chunks=chunks, inline_array=inline_array, **kwargs)


def access_parent(node: zarr.Array | zarr.Group) -> zarr.Group:
    """
    Get the parent (zarr.Group) of a Zarr array or group.
    """
    return get_parent(node)


def is_copyable(
    source_group: GroupSpec, dest_group: GroupSpec, *, strict: bool = False
) -> bool:
    """
    Check whether a Zarr group modeled by a GroupSpec `source_group` can be copied into the Zarr group modeled by GroupSpec `dest_group`.
    This entails checking that every (key, value) pair in `source_group.members` has a copyable counterpart in `dest_group.members`.
    Arrays are copyable if their shape and dtype match. Groups are copyable if their members are copyable.
    In general copyability as defined here is not symmetric for groups, because a `source_group`
    may have fewer members than `dest_group`, but if each of the shared members are copyable,
    then the source is copyable to the dest, but not vice versa.
    """

    if strict:
        if set(source_group.members.keys) != set(dest_group.members.keys):
            return False
    elif not set(source_group.members.keys()).issubset(set(dest_group.members.keys())):
        return False

    for key_source, key_dest in zip(
        source_group.members.keys(), dest_group.members.keys()
    ):
        value_source = source_group.members[key_source]
        value_dest = dest_group.members[key_dest]

        if type(value_source) != type(value_dest):
            return False
        if isinstance(value_source, ArraySpec):
            # shape and dtype are the two properties that must match for bulk copying to work.
            if not value_source.like(value_dest, include=("shape", "dtype")):
                return False
        else:
            # recurse into subgroups
            return is_copyable(value_source, value_dest, strict=strict)
    return True


def create_dataarray(
    element: zarr.Array,
    *,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataArray:
    """
    Create an xarray.DataArray from a Zarr array in an OME-NGFF hierarchy.
    """
    if coords == "auto":
        return ome_ngff.create_dataarray(
            element, use_dask=use_dask, chunks=chunks, name=name
        )

    wrapped = to_dask(element, chunks=chunks) if use_dask else element
    return DataArray(wrapped, coords=coords, attrs=attrs, name=name)


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


def to_xarray(
    element: Any,
    *,
    chunks: Literal["auto"] | tuple[int, ...] | tuple[tuple[int, ...], ...] = "auto",
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
