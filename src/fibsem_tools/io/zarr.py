import logging
import os
from os import PathLike
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Sequence, Tuple, Union
import pint

import dask.array as da
import numpy as np
import zarr
from zarr.storage import FSStore, contains_array, contains_group
from dask import bag, delayed
from distributed import Client, Lock
from toolz import concat
from xarray import DataArray
from zarr.indexing import BasicIndexer
from fibsem_tools.io.util import split_by_suffix
from fibsem_tools.metadata.transform import SpatialTransform
from pydantic_ome_ngff import Multiscale
from xarray_ome_ngff import create_coords
from fibsem_tools.io.types import JSON

ureg = pint.UnitRegistry()

# default axis order of zarr spatial metadata
# is z,y,x
ZARR_AXES_3D = ["z", "y", "x"]

# default axis order of raw n5 spatial metadata
# is x,y,z
N5_AXES_3D = ZARR_AXES_3D[::-1]
DEFAULT_ZARR_STORE = FSStore
logger = logging.getLogger(__name__)


def get_arrays(obj: Any) -> Tuple[zarr.core.Array]:
    result = ()
    if isinstance(obj, zarr.core.Array):
        result = (obj,)
    elif isinstance(obj, zarr.hierarchy.Group):
        if len(tuple(obj.arrays())) > 1:
            names, arrays = zip(*obj.arrays())
            result = tuple(concat(map(get_arrays, arrays)))
    return result


def delete_zbranch(
    branch: Union[zarr.hierarchy.Group, zarr.core.Array], compute: bool = True
):
    """
    Delete a branch (group or array) from a zarr container
    """
    if isinstance(branch, zarr.hierarchy.Group):
        return delete_zgroup(branch, compute=compute)
    elif isinstance(branch, zarr.core.Array):
        return delete_zarray(branch, compute=compute)
    else:
        raise TypeError(
            f"""
            The first argument to this function my be a zarr group or array, not 
            {type(branch)}
            """
        )


def delete_zgroup(zgroup: zarr.hierarchy.Group, compute: bool = True):
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


def delete_zarray(arr: zarr.core.Array, compute: bool = True):
    """
    Delete all the chunks in a zarr array.
    """

    if not isinstance(arr, zarr.core.Array):
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


def access_zarr(store: PathLike, path: PathLike, **kwargs) -> Any:
    if isinstance(store, Path):
        store = str(store)

    if isinstance(store, str) and kwargs.get("mode") == "w":
        store = DEFAULT_ZARR_STORE(store)

    # set default dimension separator to /
    if "shape" in kwargs and "dimension_separator" not in kwargs:
        kwargs["dimension_separator"] = "/"

    if isinstance(path, Path):
        path = str(path)

    attrs = kwargs.pop("attrs", {})
    access_mode = kwargs.pop("mode", "a")

    if access_mode == "w":
        if contains_group(store, path) or contains_array(store, path):
            # zarr is extremely slow to delete existing directories, so we do it in
            # parallel
            existing = zarr.open(store, path=path, **kwargs, mode="a")
            # todo: move this logic to methods on the stores themselves
            if isinstance(
                existing.store,
                (
                    zarr.N5Store,
                    zarr.N5FSStore,
                    zarr.DirectoryStore,
                    zarr.NestedDirectoryStore,
                ),
            ):
                url = os.path.join(existing.store.path, existing.path)
                logger.info(f"Beginning parallel deletion of chunks in {url}...")
                pre = time.time()
                delete_zbranch(existing)
                logger.info(
                    f"""
                    Completed parallel deletion of chunks in {url} in 
                    {time.time() - pre}s.
                    """
                )

    array_or_group = zarr.open(store, path=path, **kwargs, mode=access_mode)

    if access_mode != "r":
        array_or_group.attrs.update(attrs)
    return array_or_group


def access_n5(store: PathLike, path: PathLike, **kwargs) -> Any:
    store = zarr.N5FSStore(store, **kwargs.get("storage_options", {}))
    return access_zarr(store, path, **kwargs)


def zarr_to_dask(urlpath: str, chunks: Union[str, Sequence[int]], **kwargs):
    store_path, key, _ = split_by_suffix(urlpath, (".zarr",))
    arr = access_zarr(store_path, key, mode="r", **kwargs)
    if not hasattr(arr, "shape"):
        raise ValueError(f"{store_path}/{key} is not a zarr array")
    if chunks == "original":
        _chunks = arr.chunks
    else:
        _chunks = chunks
    darr = da.from_array(arr, chunks=_chunks, inline_array=True)
    return darr


def n5_to_dask(urlpath: str, chunks: Union[str, Sequence[int]], **kwargs):
    store_path, key, _ = split_by_suffix(urlpath, (".n5",))
    arr = access_n5(store_path, key, mode="r", **kwargs)
    if not hasattr(arr, "shape"):
        raise ValueError(f"{store_path}/{key} is not an n5 array")
    if chunks == "original":
        _chunks = arr.chunks
    else:
        _chunks = chunks
    darr = da.from_array(arr, chunks=_chunks, inline_array=True)
    return darr


def access_parent(node: Union[zarr.Array, zarr.Group], **kwargs):
    """
    Get the parent (zarr.Group) of a zarr array or group.
    """
    parent_path = "/".join(node.path.split("/")[:-1])
    return access_zarr(store=node.store, path=parent_path, **kwargs)


def zarr_n5_coordinate_inference(
    shape: Tuple[int, ...],
    array_attrs: Dict[str, JSON],
    group_attrs: Dict[str, JSON] = {},
    array_path: str = "",
) -> List[DataArray]:
    if (transform := array_attrs.get("transform", None)) is not None:
        coords = SpatialTransform(**transform).to_coords(shape)
    elif "pixelResolution" in array_attrs or "resolution" in array_attrs:
        input_axes = N5_AXES_3D
        output_axes = input_axes[::-1]
        translates = {ax: 0 for ax in output_axes}
        units = {ax: "nanometer" for ax in output_axes}

        if "pixelResolution" in array_attrs:
            pixelResolution = array_attrs["pixelResolution"]
            scales = dict(zip(input_axes, pixelResolution["dimensions"]))
            units = {ax: pixelResolution["unit"] for ax in input_axes}

        elif "resolution" in array_attrs:
            _scales = array_attrs["resolution"]
            scales = dict(zip(N5_AXES_3D, _scales))

        coords = [
            DataArray(
                translates[ax] + np.arange(shape[idx]) * scales[ax],
                dims=ax,
                attrs={"unit": ureg.get_name(units[ax])},
            )
            for idx, ax in enumerate(output_axes)
        ]
    elif (multiscale := group_attrs.get("multiscales", None)) is not None:
        multiscales_meta = [Multiscale(**entry) for entry in multiscale]
        transforms = []
        axes = []
        matched_multiscale = None
        matched_dataset = None
        # find the correct element in multiscales.datasets for this array
        for multi in multiscales_meta:
            for dataset in multi.datasets:
                if dataset.path == array_path:
                    matched_multiscale = multi
                    matched_dataset = dataset
        if matched_dataset is None or matched_multiscale is None:
            raise ValueError(
                f"""
            Could not find an entry referencing array {array_path} in the `multiscales`
            metadata of the parent group.
            """
            )
        else:
            transforms.extend(matched_multiscale.coordinateTransformations)
            transforms.extend(matched_dataset.coordinateTransformations)
            axes.extend(matched_multiscale.axes)
            coords = create_coords(axes, transforms, shape)
    else:
        raise ValueError("Could not infer coordinates from the supplied attributes.")

    return coords


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
