from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import zarr
from zarr.storage import BaseStore

from fibsem_tools.chunk import normalize_chunks

if TYPE_CHECKING:
    from typing import Any, Iterable, Literal, Optional, Sequence, Union

    from numpy.typing import NDArray

import dask.array as da
import fsspec
from datatree import DataTree
from numcodecs.abc import Codec
from pydantic_zarr.v2 import GroupSpec
from xarray import DataArray

from fibsem_tools.io.dat import access as access_dat
from fibsem_tools.io.dat import to_dask as dat_to_dask
from fibsem_tools.io.h5 import access as access_h5
from fibsem_tools.io.mrc import access as access_mrc
from fibsem_tools.io.mrc import to_dask as mrc_to_dask
from fibsem_tools.io.mrc import to_xarray as mrc_to_xarray
from fibsem_tools.io.n5 import access as access_n5
from fibsem_tools.io.n5.hierarchy.cosem import (
    multiscale_group as cosem_multiscale_group,
)
from fibsem_tools.io.n5.hierarchy.neuroglancer import (
    multiscale_group as neuroglancer_multiscale_group,
)
from fibsem_tools.io.tif import access as access_tif
from fibsem_tools.io.zarr import access as access_zarr
from fibsem_tools.io.zarr import to_dask as zarr_to_dask
from fibsem_tools.io.zarr import to_xarray as zarr_to_xarray
from fibsem_tools.io.zarr.hierarchy.omengff import (
    multiscale_group as ome_ngff_v04_multiscale_group,
)
from fibsem_tools.types import AccessMode, Attrs, PathLike

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
        accessor = access_zarr
    elif suffix == ".n5":
        accessor = access_n5
    elif suffix == ".h5":
        accessor = access_h5
    elif suffix in (".tif", ".tiff"):
        accessor = access_tif
    elif suffix == ".mrc":
        accessor = access_mrc
    elif suffix == ".dat":
        accessor = access_dat
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
        dasker = zarr_to_dask
    elif suffix == ".mrc":
        dasker = mrc_to_dask
    elif suffix == ".dat":
        dasker = dat_to_dask
    else:
        raise ValueError(
            f"Cannot access file with extension {suffix} as a dask array. Extensions "
            "with dask support: .zarr, .n5, .mrc, .dat"
        )
    return dasker(read(path, **kwargs), chunks)


def read_xarray(
    path: PathLike,
    chunks: Union[Literal["auto"], tuple[int, ...]] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> DataArray | DataTree:
    _, _, suffix = split_by_suffix(path, _suffixes)
    element = read(path, **kwargs)
    if suffix in (".zarr", ".n5"):
        return zarr_to_xarray(
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
        return mrc_to_xarray(
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
) -> zarr.Group:
    _arrays = tuple(a for a in arrays)
    _array_paths = tuple(p for p in array_paths)

    bad_paths = []
    for path in _array_paths:
        if len(Path(path).parts) > 1:
            bad_paths.append(path)

    if len(bad_paths):
        raise ValueError(
            "Array paths cannot be nested. The following paths violate this rule: "
            f"{bad_paths}"
        )

    group = access(group_url, mode=group_mode, attrs=group_attrs)
    a_urls = [os.path.join(group_url, name) for name in _array_paths]

    if array_attrs is None:
        _array_attrs: tuple[Attrs, ...] = ({},) * len(_arrays)
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


""" def create_dataarray(
    element: zarr.Array,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataArray:
    """ """
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

    """ """

    if name is None:
        name = element.basename

    if attrs is None:
        attrs = element.attrs.asdict()

    if coords == "auto":
        # iterate over known multiscale models
        if is_n5(element):
            creation_func = (create_n5_dataarray,)
        else:
            creation_funcs = (create_omengff_datarray,)
        # try different dataarray construction routines until one works
        exceptions: tuple[ValueError] = ()
        for func in creation_funcs:
            try:
                result = func(array=element, chunks=chunks, use_dask=use_dask)
                result.attrs.update(**attrs)
                result.name = name
                break
            except ValueError as e:
                # insert log statement here
                exceptions += (e,)
            msg = (
                f"Could not create a DataArray from {element}. "
                "The following exceptions were raised when attempting to create the dataarray: "
                f"{[str(e) for e in exceptions]}."
                "Try calling this function with coords set to a specific value instead of "
                f'"auto", or adjust the coordinate metadata in {element}'
            )
            raise ValueError(msg)
    else:
        result = DataArray(element, coords=coords, attrs=attrs, name=name)

    return result


def create_datatree(
    element: zarr.Group,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None) -> DataTree:
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
    nodes["/"] = Dataset(attrs=root_attrs)
    dtree = DataTree.from_dict(nodes, name=name)
    return dtree

def to_xarray(
    element: Any,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    coords: Any = "auto",
    name: str | None = None,
) -> DataArray | DataTree:
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
 """


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
    chunks: Union[tuple[tuple[int, ...], ...], Literal["auto"]] = "auto",
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
