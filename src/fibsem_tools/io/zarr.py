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
from toolz import concat
from xarray import DataArray, Dataset
from zarr.indexing import BasicIndexer
from fibsem_tools.io.util import PathLike
from fibsem_tools.io.xr import stt_coord
from fibsem_tools.metadata.transform import STTransform
from xarray_ome_ngff.registry import get_adapters
from zarr.errors import ReadOnlyError
from pydantic_zarr import GroupSpec, ArraySpec
from numcodecs.abc import Codec

ureg = pint.UnitRegistry()

# default axis order of zarr spatial metadata
# is z,y,x
ZARR_AXES_3D = ["z", "y", "x"]

# default axis order of raw n5 spatial metadata
# is x,y,z
N5_AXES_3D = ZARR_AXES_3D[::-1]
logger = logging.getLogger(__name__)


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


def get_arrays(obj: Union[zarr.Group, zarr.Array]) -> Tuple[zarr.Array, ...]:
    # Probably this can be removed, since zarr groups already
    # support recursively getting sub-arrays
    result = ()
    if isinstance(obj, zarr.core.Array):
        result = (obj,)
    elif isinstance(obj, zarr.hierarchy.Group):
        if len(tuple(obj.arrays())) > 1:
            names, arrays = zip(*obj.arrays())
            result = tuple(concat(map(get_arrays, arrays)))
    return result


def delete_zbranch(branch: Union[zarr.Group, zarr.Array], compute: bool = True) -> Bag:
    """
    Delete a branch (group or array) from a zarr container
    """
    if isinstance(branch, zarr.Group):
        return delete_zgroup(branch, compute=compute)
    elif isinstance(branch, zarr.Array):
        return delete_zarray(branch, compute=compute)
    else:
        raise TypeError(
            f"""
            The first argument to this function my be a zarr group or array, not 
            {type(branch)}
            """
        )


DEFAULT_ZARR_STORE = FSStorePatched
DEFAULT_N5_STORE = N5FSStorePatched


def delete_zgroup(zgroup: zarr.Group, compute: bool = True) -> None:
    """
    Delete all arrays in a zarr group
    """
    if not isinstance(zgroup, zarr.hierarchy.Group):
        raise TypeError(
            f"Cannot use the delete_zgroup function on object of type {type(zgroup)}"
        )

    arrays = get_arrays(zgroup)
    to_delete = delayed([delete_zarray(arr, compute=False) for arr in arrays])

    if compute:
        return to_delete.compute()
    else:
        return to_delete


def delete_zarray(arr: zarr.Array, compute: bool = True) -> None:
    """
    Delete all the chunks in a zarr array.
    """

    if not isinstance(arr, zarr.Array):
        raise TypeError(
            f"Cannot use the delete_zarray function on object of type {type(arr)}"
        )

    key_bag = bag.from_sequence(get_chunk_keys(arr))

    def _remove_by_keys(store: MutableMapping[str, Any], keys: List[str]) -> None:
        for key in keys:
            del store[key]

    delete_op = key_bag.map_partitions(lambda v: _remove_by_keys(arr.chunk_store, v))
    delete_op.compute()
    arr.chunk_store.rmdir(arr.path)


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


def zarr_array_from_dask(arr: da.Array) -> zarr.Array:
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
        raise ValueError(
            f"The store associated with this object has type {type(store)}, which "
            "cannot be resolved to a url"
        )


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


def get_chunk_keys(
    array: zarr.core.Array, region: slice = slice(None)
) -> Generator[str, None, None]:
    """
    Get the keys for all the chunks in a Zarr array.
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
    result = {key: Lock(key, client=client) for key in get_chunk_keys(array)}
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
                from pydantic_ome_ngff.v04 import Multiscale
            elif ngff_version == "0.5-dev":
                from pydantic_ome_ngff.latest import Multiscale
            else:
                raise ValueError(
                    "Could not resolve the version of the multiscales metadata ",
                    f"found in the group metadata {dict(group.attrs)}",
                )
        else:
            raise ValueError("Multiscales attribute was empty.")
        xarray_adapters = get_adapters(ngff_version)
        multiscales_meta = [Multiscale(**entry) for entry in multiscales]
        transforms = []
        axes = []
        matched_multiscale = None
        matched_dataset = None
        # find the correct element in multiscales.datasets for this array
        for multi in multiscales_meta:
            for dataset in multi.datasets:
                if dataset.path == array.basename:
                    matched_multiscale = multi
                    matched_dataset = dataset
        if matched_dataset is None or matched_multiscale is None:
            raise ValueError(
                f"""
            Could not find an entry referencing array {array.basename} 
            in the `multiscales` metadata of the parent group.
            """
            )
        else:
            transforms.extend(matched_multiscale.coordinateTransformations)
            transforms.extend(matched_dataset.coordinateTransformations)
            axes.extend(matched_multiscale.axes)
            coords = xarray_adapters.transforms_to_coords(axes, transforms, array.shape)
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
    attrs: Dict[str, Any] | None = None,
    name: str | None = None,
) -> DataTree:
    if coords != "auto":
        raise NotImplementedError(
            f"""
        This function does not support values of `coords` other than `auto`. 
        Got {coords}. This may change in the future.
        """
        )

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
        root_attrs = dict(element.attrs)
    else:
        root_attrs = attrs
    # insert root element
    nodes["/"] = xarray.Dataset(attrs=root_attrs)
    dtree = DataTree.from_dict(nodes, name=name)
    return dtree


@overload
def n5_spec_wrapper(spec: ArraySpec) -> ArraySpec:
    ...


@overload
def n5_spec_wrapper(spec: GroupSpec) -> GroupSpec:
    ...


def n5_spec_wrapper(spec: Union[GroupSpec, ArraySpec]) -> Union[GroupSpec, ArraySpec]:
    """
    Convert an instance of GroupSpec into one that can be materialized
    via N5Store or N5FSStore. This requires changing array compressor metadata
    and checking that the `dimension_separator` attribute is compatible with N5.

    Parameters
    ----------
    spec: Union[GroupSpec, ArraySpec]
        The spec to transform. An n5-compatible version of this spec will be generated.

    Returns
    -------
    Union[GroupSpec, ArraySpec]
    """
    if isinstance(spec, ArraySpec):
        return n5_array_wrapper(spec)
    else:
        return n5_group_wrapper(spec)


def n5_group_wrapper(spec: GroupSpec) -> GroupSpec:
    """
    Transform a GroupSpec to make it compatible with N5 storage. This function
    recursively applies itself `n5_spec_wrapper` on its members to produce an
    n5-compatible spec.

    Parameters
    ----------
    spec: GroupSpec
        The spec to transform. Only array descendants of this spec will actually be
        altered after the transformation.

    Returns
    -------
    GroupSpec
    """
    new_members = {}
    for key, member in spec.members.items():
        if hasattr(member, "shape"):
            new_members[key] = n5_array_wrapper(member)
        else:
            new_members[key] = n5_group_wrapper(member)

    return spec.__class__(attrs=spec.attrs, members=new_members)


def n5_array_wrapper(spec: ArraySpec) -> ArraySpec:
    """
    Transform an ArraySpec into one that is compatible with N5FSStore. This function
     ensures that the `dimension_separator` of the ArraySpec is ".".

    Parameters
    ----------
    spec: ArraySpec
        ArraySpec instance to be transformed.

    Returns
    -------
    ArraySpec
    """
    return spec.__class__(**{**spec.dict(), **dict(dimension_separator=".")})


def n5_array_unwrapper(spec: ArraySpec) -> ArraySpec:
    """
    Transform an ArraySpec from one parsed from an array stored in N5. This function
    applies two changes: First, the `dimension_separator` of the ArraySpec is set to
    "/", and second, the `compressor` field has some N5-specific wrapping removed.

    Parameters
    ----------
    spec: ArraySpec
        The ArraySpec to be transformed.

    Returns
    -------
    ArraySpec
    """
    new_attributes = dict(
        compressor=spec.compressor["compressor_config"], dimension_separator="/"
    )
    return spec.__class__(**{**spec.dict(), **new_attributes})


def n5_group_unwrapper(spec: GroupSpec) -> GroupSpec:
    """
    Transform a GroupSpec to remove the N5-specific attributes. Used when generating
    GroupSpec instances from Zarr groups that are stored using N5FSStore.
    This function will be applied recursively to subgroups; subarrays will be
    transformed with `n5_array_unwrapper`.

    Parameters
    ----------
    spec: GroupSpec
        The spec to be transformed.

    Returns
    -------
    GroupSpec
    """
    new_members = {}
    for key, member in spec.members.items():
        if hasattr(member, "shape"):
            new_members[key] = n5_array_unwrapper(member)
        else:
            new_members[key] = n5_group_unwrapper(member)
    return spec.__class__(attrs=spec.attrs, members=new_members)


@overload
def n5_spec_unwrapper(spec: ArraySpec) -> ArraySpec:
    ...


@overload
def n5_spec_unwrapper(spec: GroupSpec) -> GroupSpec:
    ...


def n5_spec_unwrapper(spec: Union[GroupSpec, ArraySpec]) -> Union[GroupSpec, ArraySpec]:
    """
    Transform a GroupSpec or ArraySpec to remove the N5-specific attributes.
    Used when generating GroupSpec or ArraySpec instances from Zarr groups that are
    stored using N5FSStore. If the input is an instance of GroupSpec, this
    function will be applied recursively to subgroups; subarrays will be transformed
    via `n5_array_unwrapper`. If the input is an ArraySpec, it will be transformed with
    `n5_array_unwrapper`.

    Parameters
    ----------
    spec: Union[GroupSpec, ArraySpec]
        The spec to be transformed.

    Returns
    -------
    Union[GroupSpec, ArraySpec]
    """
    if isinstance(spec, ArraySpec):
        return n5_array_unwrapper(spec)
    else:
        return n5_group_unwrapper(spec)
