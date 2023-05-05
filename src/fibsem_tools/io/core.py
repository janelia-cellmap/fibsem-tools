from __future__ import annotations
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Literal,
    Sequence,
    Tuple,
    Union,
)


from xarray import DataArray
from datatree import DataTree
from numpy.typing import NDArray

import dask.array as da
from fibsem_tools.io.util import (
    AccessMode,
    ArrayLike,
    Attrs,
    GroupLike,
    PathLike,
    split_by_suffix,
)

import fibsem_tools.io.mrc
import fibsem_tools.io.dat
import fibsem_tools.io.xr
import fibsem_tools.io.h5
import fibsem_tools.io.zarr
import fibsem_tools.io.tif

_formats = (".dat", ".mrc", ".tif", ".tiff")
_container_extensions = (".zarr", ".n5", ".h5")
_suffixes = (*_formats, *_container_extensions)


def access(
    path: PathLike,
    mode: AccessMode,
    **kwargs: Any,
) -> ArrayLike | GroupLike:
    """

    Access a variety of hierarchical array storage formats.

    Parameters
    ----------
    path: A path reference to array data or an array container.
        If `path` is a string, it is assumed to be a path, then the appropriate access
        function will be selected based on the extension of the path, and the file will
        be accessed. To access a Zarr or N5 containers, the path to the root container
        must end with .zarr or .n5

        For reading .zarr or n5 containers, this function dispatches to `zarr.open`

        For reading .dat files (Janelia-native binary image format), this function uses
        routines found in `fibsem_tools.io.dat`

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
    An array-like object or a group-like object
    """

    path_outer, path_inner, suffix = split_by_suffix(path, _suffixes)
    is_container = suffix in _container_extensions

    if suffix == ".zarr":
        accessor = fibsem_tools.io.zarr.access_zarr
    elif suffix == ".n5":
        accessor = fibsem_tools.io.zarr.access_n5
    elif suffix == ".h5":
        accessor = fibsem_tools.io.h5.access_h5
    elif suffix in (".tif", ".tiff"):
        accessor = fibsem_tools.io.tif.access
    elif suffix == ".mrc":
        accessor = fibsem_tools.io.mrc.access
    elif suffix == ".dat":
        accessor = fibsem_tools.io.dat.access
    else:
        raise ValueError(
            f"""
                Cannot access file with extension {suffix}. Try one of 
                {_suffixes}
                """
        )

    if is_container:
        return accessor(path_outer, path_inner, mode=mode, **kwargs)
    else:
        return accessor(path_outer, mode=mode, **kwargs)


def read(path: PathLike, **kwargs) -> ArrayLike | GroupLike:
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
    An array-like object or a group-like object
    zarr.hierarchy.Group

    """
    return access(path, mode="r", **kwargs)


def read_dask(
    path: PathLike,
    chunks: Union[Literal["auto"], Tuple[int, ...]] = "auto",
    **kwargs: Any,
) -> da.Array:
    """
    Create a dask array from a uri
    """
    _, _, suffix = split_by_suffix(path, _suffixes)
    if suffix in (".zarr", ".n5"):
        dasker = fibsem_tools.io.zarr.to_dask
    elif suffix == ".mrc":
        dasker = fibsem_tools.io.mrc.to_dask
    elif suffix == ".dat":
        dasker = fibsem_tools.io.dat.to_dask
    else:
        raise ValueError(
            f"""
                Cannot access file with extension {suffix} as a dask array. Extensions 
                with dask support are (".zarr", ".n5", ".mrc", and ".dat")
                """
        )
    return dasker(read(path, **kwargs), chunks)


def read_xarray(
    path: PathLike,
    chunks: Union[Literal["auto"], Tuple[int, ...]] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: Dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> DataArray | DataTree:
    _, _, suffix = split_by_suffix(path, _suffixes)
    element = read(path, **kwargs)
    if suffix in (".zarr", ".n5"):
        return fibsem_tools.io.zarr.to_xarray(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
        )
    elif suffix == ".mrc":
        # todo: support datatree semantics for mrc files, maybe by considering a folder
        # group?
        return fibsem_tools.io.mrc.to_xarray(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
        )
    else:
        raise ValueError(
            f"""
        Xarray data structures are only supported for data saved as zarr, n5, and mrc. 
        Got {type(element)}, which is not supported.
        """
        )


def create_group(
    group_url: PathLike,
    arrays: Iterable[NDArray[Any]],
    array_paths: Iterable[str],
    chunks: Sequence[int],
    group_attrs: Attrs = {},
    array_attrs: Sequence[Attrs] | None = None,
    group_mode: AccessMode = "w-",
    array_mode: AccessMode = "w-",
    **array_kwargs: Any,
) -> GroupLike:

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
