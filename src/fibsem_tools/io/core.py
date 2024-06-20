from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Hashable

import zarr
from zarr.storage import BaseStore

from fibsem_tools.chunk import normalize_chunks

if TYPE_CHECKING:
    from typing import Any, Literal, Optional, Sequence, Union

import dask.array as da
import fsspec
from datatree import DataTree
from numcodecs.abc import Codec
from pydantic_zarr.v2 import GroupSpec
from xarray import DataArray

from fibsem_tools.io import dat, h5, mrc, n5, tif
from fibsem_tools.io import zarr as zarrio

from fibsem_tools.io.n5.hierarchy.cosem import (
    model_group as cosem_multiscale_group,
)
from fibsem_tools.io.n5.hierarchy.neuroglancer import (
    model_group as neuroglancer_multiscale_group,
)
from fibsem_tools.io.zarr.hierarchy.omengff import (
    multiscale_group as ome_ngff_v04_multiscale_group,
)
from fibsem_tools.type import AccessMode, PathLike

NGFF_DEFAULT_VERSION = "0.4"
multiscale_metadata_types = ["neuroglancer", "cosem", "ome-ngff"]

_formats = (".dat", ".mrc", ".tif", ".tiff")
_container_extensions = (".zarr", ".n5", ".h5")
_suffixes = (*_formats, *_container_extensions)


def access(
    path: PathLike,
    mode: AccessMode,
    **kwargs: Any,
) -> Any:
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
        accessor = zarrio.access
    elif suffix == ".n5":
        accessor = n5.access
    elif suffix == ".h5":
        accessor = h5.access
    elif suffix in (".tif", ".tiff"):
        accessor = tif.access
    elif suffix == ".mrc":
        accessor = mrc.access
    elif suffix == ".dat":
        accessor = dat.access
    else:
        raise ValueError(
            f"Cannot access file with extension {suffix}. Try one of {_suffixes}"
        )

    if is_container:
        return accessor(path_outer, path_inner, mode=mode, **kwargs)
    else:
        return accessor(path_outer, mode=mode, **kwargs)


def read(path: PathLike, **kwargs) -> Any:
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
    chunks: Union[Literal["auto"], tuple[int, ...]] = "auto",
    **kwargs: Any,
) -> da.Array:
    """
    Create a dask array from a uri
    """
    _, _, suffix = split_by_suffix(path, _suffixes)
    if suffix in (".zarr", ".n5"):
        dasker = zarrio.to_dask
    elif suffix == ".mrc":
        dasker = mrc.to_dask
    elif suffix == ".dat":
        dasker = dat.to_dask
    else:
        raise ValueError(
            f"Cannot access file with extension {suffix} as a dask array. Extensions "
            "with dask support: .zarr, .n5, .mrc, .dat"
        )
    return dasker(read(path, **kwargs), chunks)


def read_xarray(
    path: PathLike,
    chunks: Union[Literal["auto"], tuple[int, ...]] = "auto",
    coords: Literal["auto"] | dict[Hashable, Any] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> DataArray | DataTree:
    _, _, suffix = split_by_suffix(path, _suffixes)
    element = read(path, **kwargs)
    if suffix == ".zarr":
        return zarrio.to_xarray(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
        )
    elif suffix == ".n5":
        return n5.to_xarray(
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
        return mrc.to_xarray(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
        )
    else:
        raise ValueError(
            f"Xarray data structures are only supported for data saved as zarr, n5, and mrc. "
            f"Got {type(element)}, which is not supported."
        )


def split_by_suffix(uri: PathLike, suffixes: Sequence[str]) -> tuple[str, str, str]:
    """
    Given a string and a collection of suffixes, return
    the string split at the last instance of any element of the string
    containing one of the suffixes, as well as the suffix.
    If the last element of the string bears a suffix, return the string,
    the empty string, and the suffix.
    """
    protocol: Optional[str]
    subpath: str
    protocol, subpath = fsspec.core.split_protocol(str(uri))
    if protocol is None:
        separator = os.path.sep
    else:
        separator = "/"
    parts = Path(subpath).parts
    suffixed = [Path(part).suffix in suffixes for part in parts]

    if not any(suffixed):
        msg = f"No path elements found with the suffix(es) {suffixes} in {uri}"
        raise ValueError(msg)

    index = [idx for idx, val in enumerate(suffixed) if val][-1]
    if index == (len(parts) - 1):
        pre, post = subpath, ""
    else:
        pre, post = (
            separator.join([p.strip(separator) for p in parts[: index + 1]]),
            separator.join([p.strip(separator) for p in parts[index + 1 :]]),
        )

    suffix = Path(pre).suffix
    if protocol:
        pre = f"{protocol}://{pre}"
    return pre, post, suffix


def model_multiscale_group(
    arrays: dict[str, DataArray],
    metadata_type: Literal["neuroglancer_n5", "ome-ngff", "cosem"],
    chunks: tuple[int, ...] | tuple[tuple[int, ...], ...] | Literal["auto"] = "auto",
    **kwargs,
) -> GroupSpec:
    """
    Generate a model of a multiscale group from a list of DataArrays

    Arguments
    ---------

    arrays : dict[str, DataArray]
        The arrays to store.
    metadata_type : Literal["neuroglancer_n5", "ome-ngff", "cosem"],
        The metadata flavor to use.
    chunks : Union[Tuple[Tuple[int, ...], ...], Literal["auto"]], default is "auto"
        The chunks for the arrays instances. Either an explicit collection of
        chunk sizes, one per array, or the string "auto". If `chunks` is "auto" and
        the `data` attribute of the arrays is chunked, then each stored array
        will inherit the chunks of the input arrays. If the `data` attribute
        is not chunked, then each stored array will have chunks equal to the shape of
        the input array.

    Returns
    -------

    A GroupSpec instance representing the multiscale group

    """
    _chunks = normalize_chunks(arrays.values(), chunks=chunks)

    if metadata_type == "neuroglancer":
        return neuroglancer_multiscale_group(arrays=arrays, chunks=_chunks, **kwargs)
    elif metadata_type == "cosem":
        return cosem_multiscale_group(arrays=arrays, chunks=_chunks, **kwargs)
    elif metadata_type.startswith("ome-ngff"):
        _, _, ome_ngff_version = metadata_type.partition("@")
        if ome_ngff_version in ("", "0.4"):
            return ome_ngff_v04_multiscale_group(
                arrays=arrays, transform_precision=5, chunks=_chunks, **kwargs
            )
        else:
            msg = (
                f"Metadata type {metadata_type} refers to an unsupported version of "
                "ome-ngff ({ome_ngff_version})"
            )
            raise ValueError(msg)

    else:
        msg = (
            f"Multiscale metadata type {metadata_type} is unknown."
            f"Try one of {multiscale_metadata_types}"
        )
        raise ValueError(msg)


def create_multiscale_group(
    *,
    store: BaseStore,
    path: str,
    arrays: dict[str, DataArray],
    metadata_type: Literal["neuroglancer", "cosem", "ome-ngff", "ome-ngff@0.4"],
    chunks: Union[tuple[tuple[int, ...], ...], Literal["auto"]] = "auto",
    compressor: Codec | Literal["auto"] = "auto",
    **kwargs,
) -> zarr.Group:
    group_model = model_multiscale_group(
        arrays=arrays, metadata_type=metadata_type, chunks=chunks, compressor=compressor
    )

    return group_model.to_zarr(store=store, path=path, **kwargs)
