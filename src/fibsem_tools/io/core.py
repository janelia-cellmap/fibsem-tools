from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fibsem_tools.chunk import normalize_chunks
import fibsem_tools.io.n5.core
import fibsem_tools.io.zarr.core

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import Any, Literal

    import dask.array as da
    import zarr
    from numcodecs.abc import Codec
    from pydantic_zarr.v2 import GroupSpec
    from xarray import DataArray, DataTree
    from zarr.storage import BaseStore

    from fibsem_tools.type import AccessMode, PathLike

import fsspec

from fibsem_tools.io import dat, h5, mrc, tif
from fibsem_tools.io.n5.hierarchy.cosem import (
    model_group as cosem_multiscale_group,
)
from fibsem_tools.io.n5.hierarchy.neuroglancer import (
    model_group as neuroglancer_multiscale_group,
)
from fibsem_tools.io.zarr.hierarchy.ome_ngff import (
    model_group as ome_ngff_v04_multiscale_group,
)

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
        accessor = fibsem_tools.io.zarr.access
    elif suffix == ".n5":
        accessor = fibsem_tools.io.n5.access
    elif suffix == ".h5":
        accessor = h5.access
    elif suffix in (".tif", ".tiff"):
        accessor = tif.access
    elif suffix == ".mrc":
        accessor = mrc.access
    elif suffix == ".dat":
        accessor = dat.access
    else:
        msg = f"Cannot access file with extension {suffix}. Try one of {_suffixes}"
        raise ValueError(msg)

    if is_container:
        return accessor(path_outer, path_inner, mode=mode, **kwargs)

    return accessor(path_outer, mode=mode, **kwargs)


def read(path: PathLike, **kwargs: Any) -> Any:
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

    **kwargs: Any
        Additional kwargs passed to the format-specific access function.

    Returns
    -------
    An array-like object or a group-like object
    zarr.hierarchy.Group

    """
    return access(path, mode="r", **kwargs)


def read_dask(
    path: PathLike,
    *,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    **kwargs: Any,
) -> da.Array:
    """
    Read an array from storage as a dask array.
    This function is a lightweight wrapper around `fibsem_tools.io.core.read`

    Parameters
    ----------
    path: str | Path
        The path to the array to load.
    chunks: Literal["auto"] | tuple[int, ...] = "auto"
        The chunk size to use for the dask array.
    **kwargs: Any
        Additional keyword arguments passed to `read`
    """
    _, _, suffix = split_by_suffix(path, _suffixes)
    if suffix in (".zarr", ".n5"):
        dasker = fibsem_tools.io.zarr.core.to_dask
    elif suffix == ".mrc":
        dasker = mrc.to_dask
    elif suffix == ".dat":
        dasker = dat.to_dask
    else:
        msg = (
            f"Cannot access file with extension {suffix} as a dask array. Extensions "
            "with dask support: .zarr, .n5, .mrc, .dat"
        )
        raise ValueError(msg)
    return dasker(read(path, **kwargs), chunks=chunks)


def read_xarray(
    path: PathLike,
    *,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Literal["auto"] | dict[Hashable, Any] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> DataArray | DataTree:
    """
    Read the data at `path`. If that data is an array, it will be returned
    as an `xarray.DataArray`. If it is a collection of arrays, it will be returned as
     a `Datatree`.

    Parameters
    ----------
    chunks: tuple[int, ...] | Literal["auto"]
        The chunks to use for the returned array, if `use_dask` is set to `True`.
    coords: Literal["auto"] | dict[Hashable, Any] = "auto"
        The coordinates to use for the returned array. The default value of "auto"
        results in coordinates being inferred from metadata.
    use_dask: bool = True
        Whether to wrap arrays with da.Array. Default is `True`.
    attrs: dict[str, Any] | None:
        Attributes for the returned value. The default of `None` uses the source attributes.
    name: str | None
        The name for the returned value. The default of `None` uses the source name.

    Returns
    -------
        DataArray | DataTree
    """
    _, _, suffix = split_by_suffix(path, _suffixes)
    element = read(path, **kwargs)
    if suffix == ".zarr":
        return fibsem_tools.io.zarr.core.to_xarray(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
        )
    elif suffix == ".n5":
        return fibsem_tools.io.n5.core.to_xarray(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
        )
    elif suffix == ".mrc":
        # TODO: support datatree semantics for mrc files, maybe by considering a folder
        # group?
        return mrc.to_xarray(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
        )

    msg = (
        f"Xarray data structures are only supported for data saved as zarr, n5, and mrc. "
        f"Got {type(element)}, which is not supported."
    )
    raise ValueError(msg)


def split_by_suffix(uri: PathLike, suffixes: Sequence[str]) -> tuple[str, str, str]:
    """
    Given a string and a collection of suffixes, return
    the string split at the last instance of any element of the string
    containing one of the suffixes, as well as the suffix.
    If the last element of the string bears a suffix, return the string,
    the empty string, and the suffix.
    """
    protocol: str | None
    subpath: str
    protocol, subpath = fsspec.core.split_protocol(str(uri))
    separator = os.path.sep if protocol is None else "/"
    parts = Path(subpath).parts
    suffixed = [Path(part).suffix in suffixes for part in parts]

    if not any(suffixed):
        msg = f"No path elements found with the suffix(es) {suffixes} in {uri}"
        raise ValueError(msg)

    index = [idx for idx, val in enumerate(suffixed) if val][-1]

    if index == (len(parts) - 1):
        pre, post = subpath.rstrip(separator), ""
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
    *,
    metadata_type: Literal["neuroglancer_n5", "ome-ngff", "cosem"],
    chunks: tuple[int, ...] | tuple[tuple[int, ...], ...] | Literal["auto"] = "auto",
    **kwargs: Any,
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
    **kwargs: Any
        Additional keyword arguments are passed to the groupspec constructor wrapped by this
        function.
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
        msg = (
            f"Metadata type {metadata_type} refers to an unsupported version of "
            "ome-ngff ({ome_ngff_version})"
        )
        raise ValueError(msg)

    msg = (
        f"Multiscale metadata type {metadata_type} is unknown."
        f"Try one of {multiscale_metadata_types}"
    )
    raise ValueError(msg)


def create_multiscale_group(
    store: BaseStore,
    path: str,
    arrays: dict[str, DataArray],
    *,
    metadata_type: Literal["neuroglancer", "cosem", "ome-ngff", "ome-ngff@0.4"],
    chunks: tuple[tuple[int, ...], ...] | tuple[int, ...] | Literal["auto"] = "auto",
    compressor: Codec | Literal["auto"] = "auto",
    **kwargs: Any,
) -> zarr.Group:
    """
    Create a multiscale group. This function is a light wrapper around `model_group`.

    Parameters
    ----------
    store: BaseStore
        The Zarr storage backend to use.
    path: str
        The path to the group relative to the `store`.
    arrays: dict[str, DataArray]
        The arrays to use as a template for the multiscale group
    metadata_type: Literal["neuroglancer", "cosem", "ome-ngff", "ome-ngff@0.4"]
        The flavor of metadata to use in the multiscale group.
    chunks: tuple[tuple[int, ...], ...] | tuple[int, ...] | Literal["auto"] = "auto"
        The chunks used in the arrays in the multiscale group. The default value of "auto"
        will pick chunks based on the size and datatype of the largest array in `arrays`.
    compressor: Codec | Literal["auto"] = "auto"
        The compressor to use for the arrays. The default value of "auto" will use the default
        compressor, which is Zstd.
    **kwargs: Additional keyword arguments are passed to the `to_zarr` method of GroupSpec.
    """
    group_model = model_multiscale_group(
        arrays=arrays, metadata_type=metadata_type, chunks=chunks, compressor=compressor
    )

    return group_model.to_zarr(store=store, path=path, **kwargs)
