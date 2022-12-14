from fibsem_tools.metadata.transform import SpatialTransform
from .fibsem import read_fibsem
from pathlib import Path
from numpy.typing import NDArray
from typing import (
    Union,
    Iterable,
    List,
    Optional,
    Callable,
    Dict,
    Tuple,
    Sequence,
    Any,
    Literal,
)
import dask.array as da
import os
from itertools import groupby
from collections import defaultdict
import zarr
import h5py
import mrcfile
from xarray import DataArray
from .mrc import (
    mrc_coordinate_inference,
    mrc_shape_dtype_inference,
    access_mrc,
    mrc_to_dask,
)
from .util import split_by_suffix
from .zarr import (
    access_n5,
    access_zarr,
    n5_to_dask,
    zarr_n5_coordinate_inference,
    zarr_to_dask,
)
import numcodecs
from numcodecs.abc import Codec
from enum import Enum

# encode the fact that the first axis in zarr is the z axis
_zarr_axes = {"z": 0, "y": 1, "x": 2}
# encode the fact that the first axis in n5 is the x axis
_n5_axes = {"z": 2, "y": 1, "x": 0}
_formats = (".dat", ".mrc")
_container_extensions = (".zarr", ".n5", ".h5")
_suffixes = (*_formats, *_container_extensions)

Pathlike = Union[str, Path]
defaultUnit = "nm"


class AccessMode(str, Enum):
    w = "w"
    w_minus = "w-"
    r = "r"
    r_plus = "r+"
    a = "a"


def broadcast_kwargs(**kwargs) -> Dict:
    """
    For each keyword: arg in kwargs, assert that there are only 2 types of args: sequences with length = 1
    or sequences with some length = k. Every arg with length 1 will be repeated k times, such that the return value
    is a dict of kwargs with minimum length = k.
    """
    grouped: Dict[str, List] = defaultdict(list)
    sorter = lambda v: len(v[1])
    s = sorted(kwargs.items(), key=sorter)
    for l, v in groupby(s, key=sorter):
        grouped[l].extend(v)

    assert len(grouped.keys()) <= 2
    if len(grouped.keys()) == 2:
        assert min(grouped.keys()) == 1
        output_length = max(grouped.keys())
        singletons, nonsingletons = tuple(grouped.values())
        singletons = ((k, v * output_length) for k, v in singletons)
        result = {**dict(singletons), **dict(nonsingletons)}
    else:
        result = kwargs

    return result


def access_fibsem(
    path: Union[Pathlike, Iterable[str], Iterable[Path]], mode: AccessMode
):
    if mode != "r":
        raise ValueError(
            f".dat files can only be accessed in read-only mode, not {mode}."
        )
    return read_fibsem(path)


def access_h5(
    dir_path: Pathlike, container_path: Pathlike, mode: str, **kwargs
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
    path: Union[Pathlike, Iterable[str], Iterable[Path]],
    mode: AccessMode,
    **kwargs: Dict[str, Any],
) -> Any:
    """

    Access a variety of hierarchical array storage formats.

    Parameters
    ----------
    path: A path or collection of paths to image files. If `path` is a string, it is assumed to be a path, then the appropriate access function will be
          selected based on the extension of the path, and the file will be accessed. To access a Zarr or N5 containers, the path to the root container must end with .zarr or .n5

          For reading .zarr containers, this function dispatches to `zarr.open`

          For reading n5 containers, this function uses storage routines found in `fibsem_tools.io.storage`

          For reading .dat files (Janelia-native binary image format), this function uses routines found in `fibsem_tools.io.fibsem`

          If `path` is a collection of strings,
          it is assumed that each element of the collection represents a path, and this function will return the result of calling itself on each element of the collection.

    mode: The access mode for the file. e.g. 'r' for read-only access, 'w' for writable access.

    Additional kwargs are passed to the format-specific access function.

    Returns an array-like object, a collection of array-like objects, or an instance of zarr.hierarchy.Group
    -------

    """
    if isinstance(path, (str, Path)):
        path_outer, path_inner, suffix = split_by_suffix(path, _suffixes)
        is_container = suffix in _container_extensions

        try:
            accessor = accessors[suffix]
        except KeyError:
            raise ValueError(
                f"Cannot access images with extension {suffix}. Try one of {list(accessors.keys())}"
            )

        if is_container:
            return accessor(path_outer, path_inner, mode=mode, **kwargs)
        else:
            return accessor(path_outer, mode=mode, **kwargs)

    elif isinstance(path, Iterable):
        return [access(p, mode, **kwargs) for p in path]
    else:
        raise ValueError("`path` must be a string or iterable of strings")


def read(path: Union[Pathlike, Iterable[str], Iterable[Path]], **kwargs):
    """

    Read-only access for data (arrays and groups) from a variety of hierarchical array storage formats.

    Parameters
    ----------
    path: A path or collection of paths to image files. If `path` is a string, it is assumed to be a path, then the appropriate access function will be
          selected based on the extension of the path, and the file will be accessed. To access a Zarr or N5 containers, the path to the root container must end with .zarr or .n5

          For reading .zarr containers, this function dispatches to `zarr.open`

          For reading n5 containers, this function uses storage routines found in `fibsem_tools.io.storage`

          For reading .dat files (Janelia-native binary image format), this function uses routines found in `fibsem_tools.io.fibsem`

          If `path` is a collection of strings,
          it is assumed that each element of the collection represents a path, and this function will return the result of calling itself on each element of the collection.

    Additional kwargs are passed to the format-specific access function.

    Returns an array-like object, a collection of array-like objects, or an instance of zarr.hierarchy.Group
    -------

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
    cleaned_attrs = None
    if coords == "auto":
        coords, cleaned_attrs = infer_coordinates(raw_array)
    elif isinstance(coords, SpatialTransform):
        coords = coords.to_coords(dict(zip(coords.axes, dask_array.shape)))
    if hasattr(raw_array, "attrs"):
        if not kwargs.get("attrs"):
            raw_attrs = dict(raw_array.attrs)
            if cleaned_attrs is not None:
                keys = set(raw_attrs) - set(cleaned_attrs)
                [raw_attrs.pop(key) for key in keys]
            kwargs.update({"attrs": raw_attrs})
    if kwargs.get("attrs"):
        kwargs["attrs"].update({"url": url})
    result = DataArray(dask_array, coords=coords, **kwargs)
    return result


def infer_coordinates(
    arr: Any, default_unit: str = "nm"
) -> Tuple[List[DataArray], Dict[str, Any]]:
    attrs = {}
    if isinstance(arr, zarr.core.Array):
        coords, attrs = zarr_n5_coordinate_inference(arr.shape, dict(arr.attrs))
    elif isinstance(arr, mrcfile.mrcmemmap.MrcMemmap):
        coords = mrc_coordinate_inference(arr)
    else:
        raise ValueError(
            f"No coordinate inference possible for array of type {type(arr)}"
        )
    return coords, attrs


def DataArrayFactory(arr: Any, **kwargs: Dict[str, Any]) -> DataArray:
    """
    Create an xarray.DataArray from an array-like input (e.g., zarr array, dask array). This is a very light
    wrapper around the xarray.DataArray constructor that checks for cosem/n5 metadata attributes and uses those to
    generate DataArray.coords and DataArray.dims properties; additionally, metadata about units will be inferred and
    inserted into the `attrs` kwarg if it is supplied.

    Parameters
    ----------

    arr: Array-like object (dask array or zarr array)

    """
    attrs: Optional[Dict[str, Any]]
    extra_attrs = {}

    # if we pass in a zarr array, daskify it first
    # maybe later add hdf5 support here
    if isinstance(arr, zarr.core.Array):
        source = str(Path(arr.store.path) / arr.path)
        # save the full path to the array as an attribute
        extra_attrs["source"] = source
        arr = da.from_array(arr, chunks=arr.chunks)

    if kwargs.get("attrs"):
        attrs = kwargs.get("attrs")
        out_attrs = attrs.copy()
        out_attrs.update(extra_attrs)
        kwargs["attrs"] = out_attrs
    else:
        kwargs["attrs"] = extra_attrs

    data = DataArray(arr, **kwargs)
    return data


def initialize_group(
    group_path: Pathlike,
    arrays: Sequence[NDArray[Any]],
    array_paths: Sequence[str],
    chunks: Sequence[int],
    group_attrs: Dict[str, Any] = {},
    array_attrs: Optional[Sequence[Dict[str, Any]]] = None,
    modes: Tuple[AccessMode, AccessMode] = ("w", "w"),
    **kwargs
) -> zarr.hierarchy.Group:
    group_access_mode, array_access_mode = modes
    group = access(
        group_path, mode=group_access_mode, attrs=group_attrs
    )

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
