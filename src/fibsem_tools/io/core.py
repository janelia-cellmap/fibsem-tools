import os
from os import PathLike
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import urlparse

import dask.array as da
import h5py
import mrcfile
import zarr
from numpy.typing import NDArray
from xarray import DataArray

from fibsem_tools.metadata.transform import SpatialTransform
from fibsem_tools.io.types import Attrs
from fibsem_tools.io.fibsem import read_fibsem
from fibsem_tools.io.mrc import (
    access_mrc,
    mrc_coordinate_inference,
    mrc_shape_dtype_inference,
    mrc_to_dask,
)
from fibsem_tools.io.util import split_by_suffix
from fibsem_tools.io.zarr import (
    access_n5,
    access_parent,
    access_zarr,
    n5_to_dask,
    zarr_n5_coordinate_inference,
    zarr_to_dask,
)


# encode the fact that the first axis in zarr is the z axis
_zarr_axes = {"z": 0, "y": 1, "x": 2}
# encode the fact that the first axis in n5 is the x axis
_n5_axes = {"z": 2, "y": 1, "x": 0}
_formats = (".dat", ".mrc")
_container_extensions = (".zarr", ".n5", ".h5")
_suffixes = (*_formats, *_container_extensions)


class AccessMode(str, Enum):
    w = "w"
    w_minus = "w-"
    r = "r"
    r_plus = "r+"
    a = "a"


def access_fibsem(
    path: Union[PathLike, Iterable[str], Iterable[Path]], mode: AccessMode
):
    if mode != "r":
        raise ValueError(
            f".dat files can only be accessed in read-only mode, not {mode}."
        )
    return read_fibsem(path)


def access_h5(
    dir_path: PathLike, container_path: PathLike, mode: str, **kwargs
) -> Union[h5py.Dataset, h5py.Group]:
    result = h5py.File(dir_path, mode=mode, **kwargs)
    if container_path != "":
        result = result[str(container_path)]
    return result


accessors: Dict[str, Callable[..., Any]] = {}
accessors[".dat"] = access_fibsem
accessors[".n5"] = access_n5
accessors[".zarr"] = access_zarr
accessors[".h5"] = access_h5
accessors[".mrc"] = access_mrc

daskifiers: Dict[str, Callable[..., da.core.Array]] = {}
daskifiers[".mrc"] = mrc_to_dask
daskifiers[".n5"] = n5_to_dask
daskifiers[".zarr"] = zarr_to_dask


def access(
    path: Union[PathLike, Iterable[str], Iterable[Path]],
    mode: AccessMode,
    **kwargs: Dict[str, Any],
) -> Any:
    """

    Access a variety of hierarchical array storage formats.

    Parameters
    ----------
    path: A path or collection of paths to image files.
        If `path` is a string, it is assumed to be a path, then the appropriate access
        function will be selected based on the extension of the path, and the file will
        be accessed. To access a Zarr or N5 containers, the path to the root container
        must end with .zarr or .n5

        For reading .zarr or n5 containers, this function dispatches to `zarr.open`

        For reading .dat files (Janelia-native binary image format), this function uses
        routines found in `fibsem_tools.io.fibsem`

        If `path` is a collection of strings, it is assumed that each element of the
        collection represents a path, and this function will return the result of
        calling itself on each element of the collection.

    mode: string
        The access mode for the file. e.g. 'r' for read-only access, 'w' for writable
        access.

    **kwargs: any
        Additional kwargs are passed to the format-specific access function.

    Returns
    -------
    An array-like object, a collection of array-like objects, or an instance of
    zarr.hierarchy.Group

    """
    if isinstance(path, (str, Path)):
        path_outer, path_inner, suffix = split_by_suffix(path, _suffixes)
        is_container = suffix in _container_extensions

        try:
            accessor = accessors[suffix]
        except KeyError:
            raise ValueError(
                f"""
                Cannot access images with extension {suffix}. Try one of 
                {list(accessors.keys())}
                """
            )

        if is_container:
            return accessor(path_outer, path_inner, mode=mode, **kwargs)
        else:
            return accessor(path_outer, mode=mode, **kwargs)

    elif isinstance(path, Iterable):
        return [access(p, mode, **kwargs) for p in path]
    else:
        raise ValueError("`path` must be a string or iterable of strings")


def read(path: Union[PathLike, Iterable[str], Iterable[Path]], **kwargs):
    """

    Read-only access for data (arrays and groups) from a variety of hierarchical array
    storage formats.

    Parameters
    ----------
    path: A path or collection of paths to image files.
        If `path` is a string, it is assumed to be a path, then the appropriate access
        function will be selected based on the extension of the path, and the file will
        be accessed. To access a Zarr or N5 containers, the path to the root container
        must end with .zarr or .n5

        For reading .zarr or n5 containers, this function dispatches to `zarr.open`

        For reading .dat files (Janelia-native binary image format), this function uses
        routines found in `fibsem_tools.io.fibsem`

        If `path` is a collection of strings, it is assumed that each element of the
        collection represents a path, and this function will return the result of
        calling itself on each element of the collection.

    Additional kwargs are passed to the format-specific access function.

    Returns
    -------
    An array-like object, a collection of array-like objects, or an instance of
    zarr.hierarchy.Group

    """
    return access(path, mode="r", **kwargs)


def read_dask(
    uri: str, chunks: Union[str, Tuple[int, ...]] = "auto", **kwargs: Dict[str, Any]
) -> da.core.Array:
    """
    Create a dask array from a uri
    """
    _, _, suffix = split_by_suffix(uri, _suffixes)
    return daskifiers[suffix](uri, chunks, **kwargs)


def read_xarray(
    url: str,
    chunks: Union[str, Tuple[int, ...]] = "auto",
    coords: Any = "auto",
    storage_options: Dict[str, Any] = {},
    **kwargs: Dict[str, Any],
) -> DataArray:
    """
    Create an xarray.DataArray from data found at a path.
    """
    raw_array = read(url, storage_options=storage_options)
    dask_array = read_dask(url, chunks=chunks, storage_options=storage_options)

    if coords == "auto":
        coords = infer_coordinates(raw_array)
    elif isinstance(coords, SpatialTransform):
        coords = coords.to_coords(dict(zip(coords.axes, dask_array.shape)))
    result = DataArray(dask_array, coords=coords, **kwargs)
    return result


def infer_coordinates(arr: Any, default_unit: str = "nm") -> List[DataArray]:

    if isinstance(arr, zarr.core.Array):
        coords = zarr_n5_coordinate_inference(
            shape=arr.shape,
            array_attrs=dict(arr.attrs),
            group_attrs=dict(access_parent(arr, mode="r").attrs),
            array_path=arr.basename,
        )
    elif isinstance(arr, mrcfile.mrcmemmap.MrcMemmap):
        coords = mrc_coordinate_inference(arr)
    else:
        raise ValueError(
            f"No coordinate inference possible for array of type {type(arr)}"
        )
    return coords


def initialize_group(
    group_path: PathLike,
    arrays: Sequence[NDArray[Any]],
    array_paths: Sequence[str],
    chunks: Sequence[int],
    group_attrs: Dict[str, Any] = {},
    array_attrs: Optional[Sequence[Dict[str, Any]]] = None,
    modes: Tuple[AccessMode, AccessMode] = ("w", "w"),
    **kwargs,
) -> zarr.hierarchy.Group:
    group_access_mode, array_access_mode = modes
    group = access(group_path, mode=group_access_mode, attrs=group_attrs)

    if array_attrs is None:
        _array_attrs: Tuple[Dict[str, Any], ...] = ({},) * len(arrays)
    else:
        _array_attrs = array_attrs

    for name, arr, attrs, chnks in zip(array_paths, arrays, _array_attrs, chunks):
        path = os.path.join(group.path, name)
        z_arr = zarr.open_array(
            store=group.store,
            mode=array_access_mode,
            fill_value=0,
            path=path,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=chnks,
            **kwargs,
        )
        z_arr.attrs.update(attrs)

    return group


def infer_dtype(path: str) -> str:
    fd = read(path)
    if hasattr(fd, "dtype"):
        dtype = str(fd.dtype)
    elif hasattr(fd, "data"):
        _, dtype = mrc_shape_dtype_inference(fd)
        dtype = str(dtype)
    else:
        raise ValueError(f"Cannot infer dtype of data located at {path}")
    return dtype


def create_group(
    group_url: PathLike,
    arrays: Sequence[NDArray[Any]],
    array_paths: Sequence[str],
    chunks: Sequence[int],
    group_attrs: Attrs = {},
    array_attrs: Optional[Sequence[Attrs]] = None,
    group_mode: AccessMode = "w-",
    array_mode: AccessMode = "w-",
    **array_kwargs,
) -> Tuple[str, Tuple[str, ...]]:

    bad_paths = []
    for path in array_paths:
        if len(Path(path).parts) > 1:
            bad_paths.append(path)

    if len(bad_paths):
        raise ValueError(
            f"""
            Array paths cannot be nested. The following paths violate this rule: 
            {bad_paths}
            """
        )
    protocol = urlparse(group_url).scheme
    protocol_prefix = ""
    if protocol != "":
        protocol_prefix = protocol + "://"
    group = access(group_url, mode=group_mode, attrs=group_attrs)

    if array_attrs is None:
        _array_attrs: Tuple[Attrs, ...] = ({},) * len(arrays)
    else:
        _array_attrs = array_attrs

    for idx, array in enumerate(arrays):
        name = array_paths[idx]
        path = protocol_prefix + os.path.join(group.store.path, group.path, name)
        access(
            path=path,
            mode=array_mode,
            shape=array.shape,
            dtype=array.dtype,
            chunks=chunks[idx],
            attrs=_array_attrs[idx],
            **array_kwargs,
        )
    g_url = protocol_prefix + os.path.join(group.store.path, group.path)
    a_urls = [os.path.join(g_url, name) for name in array_paths]

    return g_url, a_urls
