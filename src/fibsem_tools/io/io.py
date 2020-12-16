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
from dask.diagnostics import ProgressBar
import zarr
import h5py
from mrcfile.mrcmemmap import MrcMemmap
from xarray import DataArray
import numpy as np
from dask import bag
from .mrc import mrc_shape_dtype_inference, access_mrc, mrc_to_dask
from .util import split_path_at_suffix
from .zarr import delete_zbranch, zarr_array_from_dask
from numcodecs import GZip
import fsspec
import toolz as tz
from glob import glob

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


def access_n5(
    dir_path: Pathlike, container_path: Pathlike, **kwargs
) -> Any:
    dir_path = zarr.N5Store(dir_path)
    return access_zarr(dir_path, container_path, **kwargs)

def access_zarr(
    dir_path: Pathlike, container_path: Pathlike, **kwargs
) -> Any:
    if isinstance(dir_path, Path): 
        dir_path = str(dir_path)
    if isinstance(container_path, Path): 
        dir_path = str(dir_path)

    attrs = {}
    if 'attrs' in kwargs:
        attrs = kwargs.pop('attrs')

    # zarr is extremely slow to delete existing directories, so we do it ourselves
    if kwargs.get('mode') == 'w':
        tmp_kwargs = kwargs.copy()
        tmp_kwargs['mode'] = 'a'
        tmp = zarr.open(dir_path, path=str(container_path), **tmp_kwargs)
        delete_zbranch(tmp)
    array_or_group = zarr.open(dir_path, path=str(container_path), **kwargs)
    if kwargs.get('mode') != 'r':
        array_or_group.attrs.update(attrs)
    return array_or_group


def access_h5(
    dir_path: Pathlike, container_path: Pathlike, mode: str, **kwargs
) -> Union[h5py.Dataset, h5py.Group]:
    result = h5py.File(dir_path, mode=mode, **kwargs)
    if container_path != "":
        result = result[str(container_path)]
    return result


def access_precomputed(outer_path: Pathlike, 
                       inner_path: Pathlike, 
                       mode: str) -> Any:

    from .tensorstore import TensorStoreSpec, KVStore, parse_info
    import tensorstore as ts

    parent_dir = str(Path(outer_path).parent)
    container_dir = Path(outer_path).parts[-1]    
    kvstore = KVStore(driver='file', path=parent_dir)
    
    if mode == 'r':
        # figure out what arrays already exist within the container
        info = parse_info((Path(outer_path) / 'info').read_text())        
        scale_keys = [s.key for s in info.scales]
        if len(str(inner_path)) > 0:          
            # find the scale index corresponding to the inner path
            scale_index = scale_keys.index(str(inner_path))
        else:
            scale_index = 0

        spec = TensorStoreSpec(driver='neuroglancer_precomputed', 
                                kvstore=kvstore,
                                path=str(container_dir), 
                                scale_index=scale_index)

        result = ts.open(spec=spec.asdict(), read=True).result()
    
    elif mode == 'w':
        result = None
    
    return result

accessors: Dict[str, Callable] = {}
accessors[".dat"] = access_fibsem
accessors[".n5"] = access_n5
accessors[".zarr"] = access_zarr
accessors[".h5"] = access_h5
accessors[".mrc"] = access_mrc
accessors['.precomputed'] = access_precomputed


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
        path_outer, path_inner = split_path_at_suffix(_suffixes, path)

        # str(Path('')) => '.', which we don't want for an empty trailing path
        if str(path_inner) == ".":
            path_inner = ""

        fmt = path_outer.suffix
        is_container = fmt in _container_extensions

        try:
            accessor = accessors[fmt]
        except KeyError:
            raise ValueError(
                f"Cannot access images with extension {fmt}. Try one of {list(accessors.keys())}"
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


def infer_coordinates_3d(arr: Any, default_unit="nm") -> List[DataArray]:
    """
    Infer the coordinates and units from a 3D volume.

    """

    units: Dict[str, str]
    coords: List[DataArray]
    scaleDict: Dict[str, float]

    assert arr.ndim == 3

    # DataArray: get coords attribute directly from the data
    if hasattr(arr, "coords"):
        coords = arr.coords

    # zarr array or hdf5 array: get coords from attrs
    elif hasattr(arr, "attrs"):
        if arr.attrs.get("pixelResolution") or arr.attrs.get("resolution"):
            if pixelResolution := arr.attrs.get("pixelResolution"):
                scale: List[float] = pixelResolution["dimensions"]
                unit: str = pixelResolution["unit"]
            elif scale := arr.attrs.get("resolution"):
                unit = default_unit

            scaleDict = {k: scale[v] for k, v in _n5_axes.items()}
        else:
            scaleDict = {k: 1 for k, v in _n5_axes.items()}
            unit = default_unit

        coords = [
            DataArray(
                np.arange(arr.shape[v]) * scaleDict[k], dims=k, attrs={"units": unit}
            )
            for k, v in _zarr_axes.items()
        ]

    else:
        # check if this is a dask array constructed from a zarr array
        if isinstance(arr, da.Array) and isinstance(zarr_array_from_dask(arr), zarr.core.Array):
            arr_source: zarr.core.Array = zarr_array_from_dask(arr)
            coords = infer_coordinates_3d(arr_source)
        else:
            coords = [
                DataArray(np.arange(arr.shape[v]), dims=k, attrs={"units": defaultUnit})
                for k, v in _zarr_axes.items()
            ]

    return coords


def DataArrayFromFile(source_path: Pathlike) -> DataArray:
    if Path(source_path).suffix == '.mrc':
        arr = mrc_to_dask(source_path, chunks=(1,-1,-1)) 
    else:
        arr = read(source_path)
        arr = da.from_array(arr, chunks=arr.chunks)
        if not hasattr(arr, "chunks"):
            raise ValueError(
                f'{arr} does not have a "chunks" attribute. Is it really a distributed array-like?'
            )
    coords = infer_coordinates_3d(arr)
    data = DataArrayFactory(
        arr,
        attrs={"source": str(source_path)},
        coords=coords,
    )
    return data


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

    coords = infer_coordinates_3d(arr)
    if "coords" not in kwargs:
        kwargs.update({"coords": coords})

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
    compressor: Any=GZip(-1),
    array_attrs: Optional[Sequence[Dict[str, Any]]] = None
) -> Tuple[zarr.hierarchy.group, Tuple[zarr.core.Array]]:

    zgroup = access(os.path.join(container_path, group_path), mode='w', attrs=group_attrs)
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
        zarrays.append(access(path, 
                              shape=arr.shape, 
                              dtype=arr.dtype, 
                              chunks=chunking, 
                              compressor=compressor, 
                              attrs=attrs, 
                              mode='w'))
    return zgroup, zarrays


def infer_dtype(path: str) -> str:
    fd = read(path)
    if hasattr(fd, "dtype"):
        dtype = str(fd.dtype)
    elif hasattr(fd, "data"):
        _, dtype = mrc_shape_dtype_inference(fd)
        dtype=str(dtype)
    else:
        raise ValueError(f"Cannot infer dtype of data located at {path}")
    return dtype

    