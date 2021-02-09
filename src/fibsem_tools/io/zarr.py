from collections import defaultdict
from typing import Dict, List, Tuple, Any, Union, Sequence
from pathlib import Path
from dask import delayed
import dask.array as da
from dask.bag import from_sequence
import zarr
import os
from .util import rmtree_parallel, split_path_at_suffix
from toolz import concat
from xarray import DataArray
import numpy as np

# default axis order of zarr spatial metadata
# is z,y,x
ZARR_AXES_3D = ['z','y','x']

# default axis order of raw n5 spatial metadata
# is x,y,z
N5_AXES_3D = ZARR_AXES_3D[::-1]


def get_arrays(obj: Any) -> Tuple[zarr.core.Array]:
    result = ()
    if isinstance(obj, zarr.core.Array):
        result = (obj,)
    elif isinstance(obj, zarr.hierarchy.Group):
        if len(tuple(obj.arrays())) > 1:
            names, arrays = zip(*obj.arrays())
            result = tuple(concat(map(get_arrays, arrays)))
    return result


def delete_zbranch(branch, compute=True):
    """
    Delete a branch (group or array) from a zarr container
    """
    if isinstance(branch, zarr.hierarchy.Group):
        return delete_zgroup(branch, compute=compute)
    elif isinstance(branch, zarr.core.Array):
        return delete_zarray(branch, compute=compute)
    else:
        raise TypeError(
            f"The first argument to this function my be a zarr group or array, not {type(branch)}"
        )


def delete_zgroup(zgroup, compute=True):
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


def delete_zarray(zarray, compute=True):
    """
    Delete a zarr array.
    """

    if not isinstance(zarray, zarr.core.Array):
        raise TypeError(
            f"Cannot use the delete_zarray function on object of type {type(zarray)}"
        )

    path = os.path.join(zarray.store.path, zarray.path)
    store = zarray.store
    branch_depth = None
    if isinstance(store, zarr.N5Store) or isinstance(store, zarr.NestedDirectoryStore):
        branch_depth = 1
    elif isinstance(store, zarr.DirectoryStore):
        branch_depth = 0
    else:
        warnings.warn(
            f"Deferring to the zarr-python implementation for deleting store with type {type(store)}"
        )
        return None

    result = rmtree_parallel(path, branch_depth=branch_depth, compute=compute)
    return result


def same_compressor(arr: zarr.Array, compressor) -> bool:
    """

    Determine if the compressor associated with an array is the same as a different compressor.

    arr: A zarr array
    compressor: a Numcodecs compressor, e.g. GZip(-1)
    return: True or False, depending on whether the zarr array's compressor matches the parameters (name, level) of the
    compressor.
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
    compressor: A numcodecs compressor, e.g. GZip(-1). This will be compared with the compressor of arr.
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
    Return the zarr array that was used to create a dask array using `da.from_array(zarr_array)`
    """
    keys = tuple(arr.dask.keys())
    return arr.dask[keys[-1]]


def access_zarr(
    dir_path: Union[str, Path], container_path: Union[str, Path], **kwargs
) -> Any:
    if isinstance(dir_path, Path):
        dir_path = str(dir_path)
    if isinstance(container_path, Path):
        dir_path = str(dir_path)

    attrs = {}
    if "attrs" in kwargs:
        attrs = kwargs.pop("attrs")

    # zarr is extremely slow to delete existing directories, so we do it ourselves
    if kwargs.get("mode") == "w":
        tmp_kwargs = kwargs.copy()
        tmp_kwargs["mode"] = "a"
        tmp = zarr.open(dir_path, path=str(container_path), **tmp_kwargs)
        delete_zbranch(tmp)
    array_or_group = zarr.open(dir_path, path=str(container_path), **kwargs)
    if kwargs.get("mode") != "r":
        array_or_group.attrs.update(attrs)
    return array_or_group


def access_n5(
    dir_path: Union[str, Path], container_path: Union[str, Path], **kwargs
) -> Any:
    dir_path = zarr.N5Store(dir_path)
    return access_zarr(dir_path, container_path, **kwargs)


def zarr_to_dask(urlpath: str, chunks: Union[str, Sequence[int]]):
    store_path, key, _ = split_path_at_suffix(urlpath, ('.zarr',))
    arr = access_zarr(store_path, key, mode="r")
    if not hasattr(arr, "shape"):
        raise ValueError(f"{store_path}/{key} is not a zarr array")
    if chunks == 'original':
        _chunks = arr.chunks
    else:
        _chunks = chunks
    darr = da.from_array(arr, chunks=_chunks)
    return darr


def n5_to_dask(urlpath: str, chunks: Union[str, Sequence[int]]):
    store_path, key, _ = split_path_at_suffix(urlpath, ('.n5',))
    arr = access_n5(store_path, key, mode="r")
    if not hasattr(arr, "shape"):
        raise ValueError(f"{store_path}/{key} is not an n5 array")
    if chunks == 'original':
        _chunks = arr.chunks
    else:
        _chunks = chunks
    darr = da.from_array(arr, chunks=_chunks)
    return darr


def zarr_n5_coordinate_inference(arr: zarr.core.Array, default_unit='nm') -> List[DataArray]:
    axes: List[str] = [f'dim_{idx}' for idx in range(arr.ndim)]
    units: Dict[str, str] = {ax: default_unit for ax in axes}
    scales: Dict[str, float] = {ax: 1.0 for ax in axes}
    translates: Dict[str, float] = {ax: 0.0 for ax in axes}

    if transform_meta := arr.attrs.get('transform'):
        axes = transform_meta['axes']
        units = dict(zip(axes, transform_meta['units']))
        scales = dict(zip(axes, transform_meta['scale']))
        translates = dict(zip(axes, transform_meta['translate']))

    elif arr.attrs.get("pixelResolution") or arr.attrs.get("resolution"):
        axes = N5_AXES_3D
        translates = {ax: 0 for ax in axes}
        units = {ax: default_unit for ax in axes}
        
        if pixelResolution := arr.attrs.get("pixelResolution"):
            scales = dict(zip(axes, pixelResolution["dimensions"]))
            units: str = {ax: pixelResolution["unit"] for ax in axes}
            
        elif _scales:= arr.attrs.get("resolution"):
            scales = dict(zip(axes, _scales))
    
    coords = [
            DataArray(translates[ax] + np.arange(arr.shape[idx]) * scales[ax], dims=ax, attrs={"units": units[ax]}
            )
            for idx, ax in enumerate(ZARR_AXES_3D)]
    
    return coords