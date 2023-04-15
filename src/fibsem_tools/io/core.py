import os
import numpy.typing as npt
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import tifffile
import dask.array as da
import h5py
import mrcfile
import zarr
from numpy.typing import NDArray
from xarray import DataArray

from fibsem_tools.metadata.transform import STTransform
from fibsem_tools.io.util import Attrs, PathLike
from fibsem_tools.io.fibsem import read_fibsem
from fibsem_tools.io.mrc import (
    access_mrc,
    infer_coords as mrc_infer_coords,
    mrc_to_dask,
)
from fibsem_tools.io.util import split_by_suffix
from fibsem_tools.io.zarr import (
    access_n5,
    access_parent,
    access_zarr,
    n5_to_dask,
    infer_coords as z_infer_coords,
    zarr_to_dask,
)
from warnings import warn


_formats = (".dat", ".mrc", ".tif")
_container_extensions = (".zarr", ".n5", ".h5")
_suffixes = (*_formats, *_container_extensions)


AccessMode = Literal["w", "w-", "r", "r+", "a"]


def access_tif(
    path: PathLike, mode: Literal["r"] = "r", memmap: bool = True
) -> npt.ArrayLike:
    if mode != "r":
        raise ValueError("Tifs may only be accessed in read-only mode")

    if memmap:
        return tifffile.memmap(path)
    else:
        return tifffile.imread(path)


def access_fibsem(path: Union[PathLike, Iterable[PathLike]], mode: AccessMode):
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
accessors[".tif"] = access_tif

daskifiers: Dict[str, Callable[..., da.core.Array]] = {}
daskifiers[".mrc"] = mrc_to_dask
daskifiers[".n5"] = n5_to_dask
daskifiers[".zarr"] = zarr_to_dask


def access(
    path: Union[PathLike, Iterable[PathLike]],
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
    keep_attrs: bool = False,
    use_dask: bool = True,
    storage_options: Dict[str, Any] = {},
    **kwargs: Any,
) -> DataArray:
    """
    Create an xarray.DataArray from data found at a path.
    """
    raw_array = read(url, storage_options=storage_options)
    attrs = {}
    if keep_attrs:
        if hasattr(raw_array, "attrs"):
            attrs = dict(raw_array.attrs)
        else:
            warn(
                f"""
            The read_xarray function was invoked with the `keep_attrs` keyword argument 
            set to `True`, but the array found at the url {url} was read as an instance 
            of {type(raw_array)} which does not have an `attrs` property. This may 
            generate an error in the future.
            """
            )
    if use_dask:
        array = read_dask(url, chunks=chunks, storage_options=storage_options)
    else:
        array = raw_array

    if coords == "auto":
        coords = infer_coordinates(raw_array)

    elif isinstance(coords, STTransform):
        coords = coords.to_coords(array.shape)

    result = DataArray(array, coords=coords, attrs=attrs, **kwargs)
    return result


def infer_coordinates(arr: npt.ArrayLike) -> List[DataArray]:

    if isinstance(arr, zarr.core.Array):
        coords = z_infer_coords(
            shape=arr.shape,
            array_attrs=dict(arr.attrs),
            group_attrs=dict(access_parent(arr, mode="r").attrs),
            array_path=arr.basename,
        )
    elif isinstance(arr, mrcfile.mrcmemmap.MrcMemmap):
        coords = mrc_infer_coords(arr)
    else:
        raise ValueError(
            f"No coordinate inference possible for array of type {type(arr)}"
        )
    return coords


def create_group(
    group_url: PathLike,
    arrays: Iterable[NDArray[Any]],
    array_paths: Iterable[str],
    chunks: Sequence[int],
    group_attrs: Attrs = {},
    array_attrs: Optional[Sequence[Attrs]] = None,
    group_mode: AccessMode = "w-",
    array_mode: AccessMode = "w-",
    **array_kwargs,
) -> zarr.Group:

    _arrays = tuple(a for a in arrays)
    _array_paths = tuple(p for p in array_paths)

    bad_paths = []
    for path in _array_paths:
        if len(Path(path).parts) > 1:
            bad_paths.append(path)

    if len(bad_paths):
        raise ValueError(
            f"""
            Array paths cannot be nested. The following paths violate this rule: 
            {bad_paths}
            """
        )

    group = access(group_url, mode=group_mode, attrs=group_attrs)
    a_urls = [os.path.join(group_url, name) for name in _array_paths]

    if array_attrs is None:
        _array_attrs: Tuple[Attrs, ...] = ({},) * len(_arrays)
    else:
        _array_attrs = array_attrs

    for idx, vals in enumerate(zip(_arrays, a_urls, _array_attrs)):
        array, path, attrs = vals
        access(
            path=path,
            mode=array_mode,
            shape=array.shape,
            dtype=array.dtype,
            chunks=chunks[idx],
            attrs=attrs,
            **array_kwargs,
        )

    return group
