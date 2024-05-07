from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    MutableMapping,
    Sequence,
    Tuple,
    Union,
    overload,
)
from datatree import DataTree
import pint

import dask.array as da
from dask.bag import Bag
import numpy as np
import xarray
from dask.base import tokenize
import zarr
from zarr.storage import FSStore, BaseStore
from dask import bag, delayed
from distributed import Client, Lock
from xarray import DataArray, Dataset
from zarr.indexing import BasicIndexer
from fibsem_tools.io.util import PathLike
from fibsem_tools.io.xr import stt_coord
from fibsem_tools.metadata.transform import STTransform
from zarr.errors import ReadOnlyError
from pydantic_zarr.v2 import GroupSpec, ArraySpec
from numcodecs.abc import Codec
from operator import delitem
from zarr.errors import PathNotFoundError
from cellmap_schemas.n5_wrap import n5_spec_unwrapper, n5_spec_wrapper

ureg = pint.UnitRegistry()

# default axis order of zarr spatial metadata
# is z,y,x
ZARR_AXES_3D = ["z", "y", "x"]

# default axis order of raw n5 spatial metadata
# is x,y,z
N5_AXES_3D = ZARR_AXES_3D[::-1]
logger = logging.getLogger(__name__)


def parse_url(url: str) -> Tuple[str, str]:
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
    Patch delitems to delete "blind", i.e. without checking if to-be-deleted keys exist.
    This is temporary and should be removed when
    https://github.com/zarr-developers/zarr-python/issues/1336
    is resolved.
    """

    def delitems(self, keys: Sequence[str]) -> None:
        if self.mode == "r":
            raise ReadOnlyError()
        try:  # should much faster
            nkeys = [self._normalize_key(key) for key in keys]
            # rm errors if you pass an empty collection
            self.map.delitems(nkeys)
        except FileNotFoundError:
            nkeys = [self._normalize_key(key) for key in keys if key in self]
            # rm errors if you pass an empty collection
            if len(nkeys) > 0:
                self.map.delitems(nkeys)


class N5FSStorePatched(zarr.N5FSStore):
    """
    Patch delitems to delete "blindly", i.e. without checking if to-be-deleted keys exist.
    This is temporary and should be removed when
    https://github.com/zarr-developers/zarr-python/issues/1336 is resolved.
    """

    def delitems(self, keys: Sequence[str]) -> None:
        if self.mode == "r":
            raise ReadOnlyError()
        try:  # should much faster
            nkeys = [self._normalize_key(key) for key in keys]
            # rm errors if you pass an empty collection
            self.map.delitems(nkeys)
        except FileNotFoundError:
            nkeys = [self._normalize_key(key) for key in keys if key in self]
            # rm errors if you pass an empty collection
            if len(nkeys) > 0:
                self.map.delitems(nkeys)


def delete_node(branch: Union[zarr.Group, zarr.Array], compute: bool = True) -> Bag:
    """
    Delete a branch (group or array) from a zarr container
    """
    if isinstance(branch, zarr.Group):
        return delete_group(branch, compute=compute)
    elif isinstance(branch, zarr.Array):
        return delete_array(branch, compute=compute)
    else:
        msg = f"The first argument to this function my be a zarr group or array, not {type(branch)}"
        raise TypeError(msg)


DEFAULT_ZARR_STORE = FSStorePatched
DEFAULT_N5_STORE = N5FSStorePatched


def delete_group(group: zarr.Group, compute: bool = True) -> None:
    """
    Delete a zarr group, and everything it contains
    """
    if not isinstance(group, zarr.hierarchy.Group):
        raise TypeError(
            f"Cannot use the delete_zgroup function on object of type {type(group)}"
        )

    _, arrays = zip(*group.arrays(recurse=True))
    to_delete = delayed([delete_array(arr, compute=False) for arr in arrays])

    if compute:
        result = to_delete.compute()
        to_delete.store.rmdir(group.path)
    else:
        result = to_delete
    return result


def delete_array(array: zarr.Array, compute: bool = True) -> None:
    """
    Delete all the chunks in a zarr array.
    """

    if not isinstance(array, zarr.Array):
        raise TypeError(
            f"Cannot use the delete_zarray function on object of type {type(array)}"
        )

    key_bag = bag.from_sequence(chunk_keys(array))
    delete_op = key_bag.map(lambda v: delitem(array.chunk_store, v))
    if compute:
        result = delete_op.compute()
        array.chunk_store.rmdir(array.path)
    else:
        result = delete_op

    return result


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


def same_array_props(
    arr: zarr.Array, shape: Tuple[int], dtype: str, compressor: Any, chunks: Tuple[int]
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
    maybe_array: Union[zarr.Array, Tuple[Any, zarr.Array, Tuple[slice]]] = tuple(
        arr.dask.values()
    )[0]
    if isinstance(maybe_array, zarr.Array):
        return maybe_array
    else:
        return maybe_array[1]


def get_url(node: Union[zarr.Group, zarr.Array]) -> str:
    store = node.store
    if hasattr(store, "path"):
        if hasattr(store, "fs"):
            if isinstance(store.fs.protocol, Sequence):
                protocol = store.fs.protocol[0]
            else:
                protocol = store.fs.protocol
        else:
            protocol = "file"

        # fsstore keeps the protocol in the path, but not s3store
        if "://" in store.path:
            store_path = store.path.split("://")[-1]
        else:
            store_path = store.path
        return f"{protocol}://{os.path.join(store_path, node.path)}"
    else:
        msg = (
            f"The store associated with this object has type {type(store)}, which "
            "cannot be resolved to a url"
        )
        raise ValueError(msg)


def get_store(path: PathLike) -> zarr.storage.BaseStore:
    if isinstance(path, Path):
        path = str(path)

    return DEFAULT_ZARR_STORE(path)


def access_zarr(
    store: Union[PathLike, BaseStore], path: PathLike, **kwargs: Any
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


def access_n5(store: PathLike, path: PathLike, **kwargs: Any) -> Any:
    store = DEFAULT_N5_STORE(store, **kwargs.get("storage_options", {}))
    return access_zarr(store, path, **kwargs)


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


def access_parent(node: Union[zarr.Array, zarr.Group], **kwargs: Any) -> zarr.Group:
    """
    Get the parent (zarr.Group) of a Zarr array or group.
    """
    parent_path = "/".join(node.path.split("/")[:-1])
    return access_zarr(store=node.store, path=parent_path, **kwargs)


def is_n5(array: zarr.core.Array) -> bool:
    """
    Check if a Zarr array is backed by N5 storage.
    """
    return isinstance(array.store, (zarr.N5Store, zarr.N5FSStore))


def chunk_keys(
    array: zarr.core.Array, region: Union[slice, Tuple[slice, ...]] = slice(None)
) -> Generator[str, None, None]:
    """
    Get the keys for all the chunks in a Zarr array as a generator of strings.

    Parameters
    ----------
    array: zarr.core.Array
        The zarr array to get the chunk keys from
    region: Union[slice, Tuple[slice, ...]]
        The region in the zarr array get chunks keys from. Defaults to `slice(None)`, which
        will result in all the chunk keys being returned.
    Returns
    -------
    Generator[str, None, None]

    """
    indexer = BasicIndexer(region, array)
    chunk_coords = (idx.chunk_coords for idx in indexer)
    keys = (array._chunk_key(cc) for cc in chunk_coords)
    return keys


def chunk_grid_shape(
    array_shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Get the shape of the chunk grid of a Zarr array.
    """
    return tuple(np.ceil(np.divide(array_shape, chunk_shape)).astype("int").tolist())


class ChunkLock:
    def __init__(self, array: zarr.core.Array, client: Client):
        self._locks = get_chunklock(array, client)
        # from the perspective of a zarr array, metadata has this key regardless of the
        # location on storage. unfortunately, the synchronizer does not get access to
        # the indirection provided by the the store class.

        array_attrs_key = f"{array.path}/.zarray"
        if is_n5(array):
            attrs_path = f"{array.path}/attributes.json"
        else:
            attrs_path = array_attrs_key
        self._locks[array_attrs_key] = Lock(attrs_path, client=client)

    def __getitem__(self, key: str):
        return self._locks[key]


def get_chunklock(array: zarr.core.Array, client: Client) -> Dict[str, Lock]:
    result = {key: Lock(key, client=client) for key in chunk_keys(array)}
    return result


def lock_array(array: zarr.core.Array, client: Client) -> zarr.core.Array:
    lock = ChunkLock(array, client)
    locked_array = zarr.open(
        store=array.store, path=array.path, synchronizer=lock, mode="a"
    )
    return locked_array


def are_chunks_aligned(
    source_chunks: Tuple[int, ...], dest_chunks: Tuple[int, ...]
) -> bool:
    assert len(source_chunks) == len(dest_chunks)
    return all(
        s_chunk % d_chunk == 0 for s_chunk, d_chunk in zip(source_chunks, dest_chunks)
    )


def infer_coords(array: zarr.Array) -> List[DataArray]:
    """
    Attempt to infer coordinate data from a Zarr array. This function loops over potential
    metadata schemes, trying each until one works.
    """
    group = access_parent(array, mode="r")

    if (transform := array.attrs.get("transform", None)) is not None:
        coords = STTransform(**transform).to_coords(array.shape)

    elif "pixelResolution" in array.attrs or "resolution" in array.attrs:
        input_axes = N5_AXES_3D
        output_axes = input_axes[::-1]
        translates = {ax: 0 for ax in output_axes}
        units = {ax: "nanometer" for ax in output_axes}

        if "pixelResolution" in array.attrs:
            pixelResolution = array.attrs["pixelResolution"]
            scales = dict(zip(input_axes, pixelResolution["dimensions"]))
            units = {ax: pixelResolution["unit"] for ax in input_axes}

        elif "resolution" in array.attrs:
            _scales = array.attrs["resolution"]
            scales = dict(zip(N5_AXES_3D, _scales))

        coords = [
            stt_coord(array.shape[idx], ax, scales[ax], translates[ax], units[ax])
            for idx, ax in enumerate(output_axes)
        ]
    elif (multiscales := group.attrs.get("multiscales", None)) is not None:
        if len(multiscales) > 0:
            multiscale = multiscales[0]
            ngff_version = multiscale.get("version", None)
            if ngff_version == "0.4":
                from xarray_ome_ngff.v04.multiscale import read_array
                from xarray_ome_ngff.array_wrap import DaskArrayWrapper

                coords = read_array(
                    array=array, array_wrapper=DaskArrayWrapper(chunks="auto")
                ).coords
            else:
                raise ValueError(
                    "Could not resolve the version of the multiscales metadata ",
                    f"found in the group metadata {dict(group.attrs)}",
                )
        else:
            raise ValueError("Multiscales attribute was empty.")
    else:
        raise ValueError(
            f"Could not infer coordinates for {array.path}, located at {array.store.path}."
        )

    return coords


@overload
def to_xarray(
    element: zarr.Array,
    chunks: Literal["auto"] | Tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: Dict[str, Any] | None = None,
    coords: Any = "auto",
    name: str | None = None,
) -> DataArray:
    ...


@overload
def to_xarray(
    element: zarr.Group,
    chunks: Literal["auto"] | Tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: Dict[str, Any] | None = None,
    coords: Any = "auto",
    name: str | None = None,
) -> DataTree:
    ...


def to_xarray(
    element: zarr.Array | zarr.Group,
    chunks: Literal["auto"] | Tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: Dict[str, Any] | None = None,
    coords: Any = "auto",
    name: str | None = None,
) -> Union[DataArray, DataTree]:
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


def create_dataarray(
    element: zarr.Array,
    chunks: Literal["auto"] | Tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: Dict[str, Any] | None = None,
    name: str | None = None,
) -> DataArray:
    """
    Create an xarray.DataArray from a zarr array.

    Parameters
    ----------

    element : zarr.Array

    chunks : Literal['auto'] | tuple[int, ...] = "auto"
        The chunks for the array, if `use_dask` is set to `True`

    coords : Any, default is "auto"
        Coordinates for the data. If `coords` is "auto", then `infer_coords` will be called on
        `element` to read the coordinates based on metadata. Otherwise, `coords` should be
        a valid argument to the `coords` keyword argument in the `DataArray` constructor.

    use_dask : bool
        Whether to wrap `element` in a Dask array before creating the DataArray.

    attrs : dict[str, Any] | None
        Attributes for the `DataArray`. if None, then attributes will be inferred from the `attrs`
        property of `element`.

    name : str | None
        Name for the `DataArray` (and the underlying Dask array, if `to_dask` is `True`)

    Returns
    -------

    xarray.DataArray

    """

    if name is None:
        name = element.basename

    if coords == "auto":
        coords = infer_coords(element)

    # todo: fix this unfortunate structure
    elif hasattr(coords, "to_coords"):
        coords = coords.to_coords(element.shape)

    if attrs is None:
        attrs = dict(element.attrs)

    if use_dask:
        element = to_dask(element, chunks=chunks)

    result = xarray.DataArray(element, coords=coords, attrs=attrs, name=name)
    return result


def create_datatree(
    element: zarr.Group,
    chunks: Union[Literal["auto"], Tuple[int, ...]] = "auto",
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

    nodes: MutableMapping[str, Dataset | DataArray | DataTree | None] = {
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
    nodes["/"] = xarray.Dataset(attrs=root_attrs)
    dtree = DataTree.from_dict(nodes, name=name)
    return dtree


def ensure_spec(
    spec: Union[GroupSpec, ArraySpec], store, path: str
) -> Union[GroupSpec, ArraySpec]:
    """
    Ensure that a zarr storage backend has the structure specified by a spec
    """
    try:
        existing_node = zarr.open(store=store, path=path, mode="r")
        existing_spec = GroupSpec.from_zarr(existing_node)
        if isinstance(store, zarr.N5FSStore):
            existing_spec = n5_spec_unwrapper(existing_spec)
        if existing_spec == spec:
            node = zarr.open(store=store, path=path)
        else:
            # todo: make a specific exception for this
            raise PathNotFoundError(path)

    # Neither a zarr Array or Group was found at that path
    # so we can safely instantiate the spec
    except PathNotFoundError:
        if isinstance(store, zarr.N5FSStore):
            spec = n5_spec_wrapper(spec)
        node = spec.to_zarr(store, path=path)
    return node


def copyable(source_group: GroupSpec, dest_group: GroupSpec, strict: bool = False):
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
    else:
        # extra members are allowed in the destination group
        if not set(source_group.members.keys()).issubset(
            set(dest_group.members.keys())
        ):
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
            if value_source.shape != value_dest.shape:
                return False
            if value_source.dtype != value_dest.dtype:
                return False
        else:
            # recurse into subgroups
            return copyable(value_source, value_dest, strict=strict)
    return True
