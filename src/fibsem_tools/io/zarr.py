from __future__ import annotations
import logging
import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Sequence, Tuple, Union
from datatree import DataTree
import pint

import dask.array as da
import numpy as np
import xarray
import zarr
from zarr.storage import FSStore, contains_array, contains_group, BaseStore
from dask import bag, delayed
from distributed import Client, Lock
from toolz import concat
from xarray import DataArray
from zarr.indexing import BasicIndexer
from fibsem_tools.io.xr import stt_coord
from fibsem_tools.metadata.transform import STTransform
from xarray_ome_ngff.registry import get_adapters
from zarr.errors import ReadOnlyError

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

    def delitems(self, keys):
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
    Patch delitems to delete "blind", i.e. without checking if to-be-deleted keys exist.
    This is temporary and should be removed when
    https://github.com/zarr-developers/zarr-python/issues/1336
    is resolved.
    """

    def delitems(self, keys):
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


def get_arrays(obj: Any) -> Tuple[zarr.Array]:
    result = ()
    if isinstance(obj, zarr.core.Array):
        result = (obj,)
    elif isinstance(obj, zarr.hierarchy.Group):
        if len(tuple(obj.arrays())) > 1:
            names, arrays = zip(*obj.arrays())
            result = tuple(concat(map(get_arrays, arrays)))
    return result


def delete_zbranch(branch: Union[zarr.Group, zarr.Array], compute: bool = True):
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


def delete_zgroup(zgroup: zarr.Group, compute: bool = True):
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


def delete_zarray(arr: zarr.Array, compute: bool = True):
    """
    Delete all the chunks in a zarr array.
    """

    if not isinstance(arr, zarr.Array):
        raise TypeError(
            f"Cannot use the delete_zarray function on object of type {type(arr)}"
        )

    key_bag = bag.from_sequence(get_chunk_keys(arr))

    def _remove_by_keys(store, keys: List[str]):
        for key in keys:
            del store[key]

    delete_op = key_bag.map_partitions(lambda v: _remove_by_keys(arr.chunk_store, v))
    delete_op.compute()
    arr.chunk_store.rmdir(arr.path)


def same_compressor(arr: zarr.Array, compressor) -> bool:
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


def zarr_array_from_dask(arr: Any) -> Any:
    """
    Return the zarr array that was used to create a dask array using
    `da.from_array(zarr_array)`
    """
    keys = tuple(arr.dask.keys())
    return arr.dask[keys[-1]]


def get_url(node: Union[zarr.Group, zarr.Array]):
    store = node.store
    if hasattr(store, "path"):
        if hasattr(store, "fs"):
            if isinstance(store.fs.protocol, list):
                protocol = store.fs.protocol[0]
            else:
                protocol = store.fs.protocol
        else:
            protocol = "file"
        return f"{protocol}://{os.path.join(node.store.path, node.path)}"
    else:
        raise ValueError(
            f"""
        The store associated with this object has type {type(store)}, which 
        cannot be resolved to a url"""
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
    arr: zarr.Array, chunks: Literal["auto"] | Sequence[int], name: str | None = None
):
    darr = da.from_array(arr, chunks=chunks, inline_array=True, name=name)
    return darr


def access_parent(node: Union[zarr.Array, zarr.Group], **kwargs):
    """
    Get the parent (zarr.Group) of a zarr array or group.
    """
    parent_path = "/".join(node.path.split("/")[:-1])
    return access_zarr(store=node.store, path=parent_path, **kwargs)


def is_n5(array: zarr.core.Array) -> bool:
    return isinstance(array.store, (zarr.N5Store, zarr.N5FSStore))


def get_chunk_keys(
    array: zarr.core.Array, region: slice = slice(None)
) -> Generator[str, None, None]:
    indexer = BasicIndexer(region, array)
    chunk_coords = (idx.chunk_coords for idx in indexer)
    keys = (array._chunk_key(cc) for cc in chunk_coords)
    return keys


def chunk_grid_shape(
    array_shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
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

    def __getitem__(self, key):
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
                    f"""
                    Could not resolve the version of the multiscales metadata 
                    found in the group metadata {dict(group.attrs)}
                    """
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
        raise ValueError("Could not infer coordinates from the supplied attributes.")

    return coords


def to_xarray(
    element: zarr.Array | zarr.Group,
    chunks: Literal["auto"] | Tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: Dict[str, Any] | None = None,
    coords: Any = "auto",
    name: str | None = None,
):
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
            f"""
        This function only accepts instances of zarr.Group and zarr.Array. 
        Got {type(element)} instead.
        """
        )


def create_dataarray(
    element: zarr.Array,
    chunks: Literal["auto"] | Tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: Dict[str, Any] | None = None,
    name: str | None = None,
) -> xarray.DataArray:
    """
    Create an xarray.DataArray from a zarr array.
    """

    if name is None:
        name = element.basename

    if coords == "auto":
        coords = infer_coords(element)

    # todo: fix this unfortunate structure
    elif hasattr(coords, "to_coords"):
        coords = coords.to_coords(element.shape)

    if use_dask:
        if attrs is None:
            attrs = dict(element.attrs)
        element = to_dask(element, chunks=chunks, name=name)

    result = xarray.DataArray(element, coords=coords, attrs=attrs, name=name)
    return result


def create_datatree(
    element: zarr.Group,
    chunks: Union[str, Tuple[int, ...]] = "auto",
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

    nodes = {
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
