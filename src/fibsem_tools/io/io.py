from .fibsem import read_fibsem
from pathlib import Path
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
)
from dask import delayed
import dask.array as da
import os
from itertools import groupby
from collections import defaultdict
import zarr
import h5py
import mrcfile
from xarray import DataArray
import numpy as np
from dask import bag
from .mrc import mrc_coordinate_inference, mrc_shape_dtype_inference, access_mrc, mrc_to_dask
from .util import split_path_at_suffix
from .zarr import zarr_array_from_dask, access_n5, access_zarr, n5_to_dask, zarr_n5_coordinate_inference, zarr_to_dask
from .tensorstore import access_precomputed, precomputed_to_dask
from numcodecs import GZip
import fsspec
import toolz as tz
from glob import glob
import distributed

# encode the fact that the first axis in zarr is the z axis
_zarr_axes = {"z": 0, "y": 1, "x": 2}
# encode the fact that the first axis in n5 is the x axis
_n5_axes = {"z": 2, "y": 1, "x": 0}
_formats = (".dat", ".mrc")
_container_extensions = (".zarr", ".n5", ".h5", ".precomputed")
_suffixes = (*_formats, *_container_extensions)

Pathlike = Union[str, Path]
defaultUnit = "nm"


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


def access_fibsem(path: Union[Pathlike, Iterable[str], Iterable[Path]], mode: str):
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
accessors[".precomputed"] = access_precomputed

daskifiers: Dict[str, Callable[..., da.core.Array]] = {}
daskifiers[".mrc"] = mrc_to_dask
daskifiers[".n5"] = n5_to_dask
daskifiers[".zarr"] = zarr_to_dask
daskifiers[".precomputed"] = precomputed_to_dask


def access(
    path: Union[Pathlike, Iterable[str], Iterable[Path]],
    mode: str,
    **kwargs,
) -> Any:
    """

    Access data from a variety of array storage formats.

    Parameters
    ----------
    path: A path or collection of paths to image files. If `path` is a string, then the appropriate reader will be
          selected based on the extension of the path, and the file will be read. If `path` is a collection of strings,
          it is assumed that each string is a path to an image and each will be read sequentially.

    lazy: A boolean, defaults to False. If True, this function returns the native file reader wrapped by
    dask.delayed. This is advantageous for distributed computing.

    mode: The access mode for the file. e.g. 'r' for read-only access.

    Returns an array-like object, a collection of array-like objects, a chunked store, or
    a dask.delayed object.
    -------

    """
    if isinstance(path, (str, Path)):
        path_outer, path_inner, suffix = split_path_at_suffix(path, _suffixes)
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

    Access data on disk with read-only permissions

    Parameters
    ----------
    path: A path or collection of paths to image files. If `path` is a string, then the appropriate image reader will be
          selected based on the extension of the path, and the file will be read. If `path` is a collection of strings,
          it is assumed that each string is a path to an image and each will be read sequentially.

    lazy: A boolean, defaults to False. If True, this function returns the native file reader wrapped by
    dask.delayed. This is advantageous for distributed computing.

    Returns an array-like object, a collection of array-like objects, a chunked store, or
    a dask.delayed object.
    -------

    """
    return access(path, mode="r", **kwargs)


def read_dask(urlpath: str, chunks='auto', **kwargs) -> da.core.Array:
    """
    Create a dask array from a path
    """
    path_outer, path_inner, suffix = split_path_at_suffix(urlpath, _suffixes)
    return daskifiers[suffix](urlpath, chunks, **kwargs)


def read_xarray(urlpath: str, chunks: Union[str, Tuple[int,...]]='auto', coords: Any='auto', **kwargs) -> DataArray:
    """
    Create an xarray.DataArray from data found at a path. 
    """
    raw_array = read(urlpath)
    if hasattr(raw_array, 'attrs'):
        if not kwargs.get('attrs'):
            kwargs.update({'attrs': dict(raw_array.attrs)}) 
    if kwargs.get('attrs'):
        kwargs['attrs'].update({'urlpath': urlpath})
    dask_array = read_dask(urlpath, chunks=chunks)
    if coords == 'auto':
        coords = infer_coordinates(raw_array)    
    result = DataArray(dask_array, coords=coords, **kwargs)
    return result


def infer_coordinates(arr: Any, default_unit: str="nm") -> List[DataArray]:
    if isinstance(arr, zarr.core.Array):
        coords = zarr_n5_coordinate_inference(arr)
    elif isinstance(arr, mrcfile.mrcmemmap.MrcMemmap):
        coords = mrc_coordinate_inference(arr)
    else:
        raise ValueError(f'No coordinate inference possible for array of type {type(arr)}')
    return coords


def DataArrayFactory(arr: Any, **kwargs) -> DataArray:
    """
    Create an xarray.DataArray from an array-like input (e.g., zarr array, dask array). This is a very light
    wrapper around the xarray.DataArray constructor that checks for cosem/n5 metadata attributes and uses those to
    generate DataArray.coords and DataArray.dims properties; additionally, metadata about units will be inferred and
    inserted into the `attrs` kwarg if it is supplied.

    Parameters
    ----------

    arr: Array-like object (dask array or zarr array)

    """
    attrs: Optional[Dict]
    extra_attrs = {}

    # if we pass in a zarr array, daskify it first
    # maybe later add hdf5 support here
    if isinstance(arr, zarr.core.Array):
        source = str(Path(arr.store.path) / arr.path)
        # save the full path to the array as an attribute
        extra_attrs["source"] = source
        arr = da.from_array(arr, chunks=arr.chunks)

    if attrs := kwargs.get("attrs"):
        out_attrs = attrs.copy()
        out_attrs.update(extra_attrs)
        kwargs["attrs"] = out_attrs
    else:
        kwargs["attrs"] = extra_attrs

    data = DataArray(arr, **kwargs)
    return data


def populate_group(
    container_path: Pathlike,
    group_path: Pathlike,
    arrays: Sequence[DataArray],
    array_paths: Sequence[str],
    chunks: Sequence[int],
    group_attrs: Dict[str, Any] = {},
    compressor: Any = GZip(-1),
    array_attrs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[zarr.hierarchy.group, Tuple[zarr.core.Array]]:

    zgroup = access(
        os.path.join(container_path, group_path), mode="w", attrs=group_attrs
    )
    zarrays = []
    if array_attrs == None:
        _array_attrs = ({},) * len(arrays)
    else:
        _array_attrs = array_attrs

    for idx, arr in enumerate(arrays):
        path = os.path.join(container_path, group_path, array_paths[idx])
        chunking = chunks[idx]
        compressor = compressor
        attrs = _array_attrs[idx]
        zarrays.append(
            access(
                path,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=chunking,
                compressor=compressor,
                attrs=attrs,
                mode="w",
            )
        )
    return zgroup, zarrays


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
