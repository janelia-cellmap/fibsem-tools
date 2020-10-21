from dataclasses import asdict
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
from shutil import rmtree
from glob import glob
from itertools import groupby
from collections import defaultdict
from dask.diagnostics import ProgressBar
import zarr
import h5py
import dask
from mrcfile.mrcmemmap import MrcMemmap
from xarray import DataArray
from xarray.core.coordinates import DataArrayCoordinates
import numpy as np
from dask import bag
from zarr.core import Array as ZarrArray
from zarr.hierarchy import Group as ZarrGroup
from .mrc import mrc_shape_dtype_inference, access_mrc, mrc_to_dask
from numcodecs import GZip

# encode the fact that the first axis in zarr is the z axis
_zarr_axes = {"z": 0, "y": 1, "x": 2}
# encode the fact that the first axis in n5 is the x axis
_n5_axes = {"z": 2, "y": 1, "x": 0}
_formats = (".dat", ".mrc")
_container_extensions = (".zarr", ".n5", ".h5", ".precomputed")
_suffixes = (*_formats, *_container_extensions)

Pathlike = Union[str, Path]
Arraylike = Union[zarr.core.Array, da.Array, DataArray, np.array]
ArraySources = Union[
    List[dask.delayed], zarr.core.Array, zarr.hierarchy.Group, h5py.Dataset, h5py.Group, np.array
]

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


def split_path_at_suffix(
    upper_path: Pathlike, lower_path: Pathlike = "", suffixes: tuple = _suffixes
) -> List[Path]:
    """
    Recursively climb a path, checking at each level of the path whether the tail of the path represents a directory
    with a container extension. Returns the path broken at the level where a container is found.  
    """
    upper, lower = Path(upper_path), Path(lower_path)

    if upper.suffix in suffixes:
        result = [upper, lower]
    else:
        if len(upper.parts) >= 2:
            result = split_path_at_suffix(
                Path(*upper.parts[:-1]), Path(upper.parts[-1], lower), suffixes
            )
        else:
            raise ValueError(
                f"Could not find any suffixes matching {suffixes} in {upper / lower}"
            )

    return result


def access_fibsem(path: Union[Pathlike, Iterable[str], Iterable[Path]], mode: str):
    if mode != "r":
        raise ValueError(
            f".dat files can only be accessed in read-only mode, not {mode}."
        )
    return read_fibsem(path)


def access_n5(
    dir_path: Pathlike, container_path: Pathlike, **kwargs
) -> Union[zarr.core.Array, zarr.hierarchy.Group]:
    return zarr.open(zarr.N5Store(dir_path), path=container_path, **kwargs)


def access_zarr(
    dir_path: Pathlike, container_path: Pathlike, **kwargs
) -> Union[zarr.core.Array, zarr.hierarchy.Group]:
    return zarr.open(str(dir_path), path=str(container_path), **kwargs)


def access_h5(
    dir_path: Pathlike, container_path: Pathlike, mode: str, **kwargs
) -> Union[h5py.Dataset, h5py.Group]:
    result = h5py.File(dir_path, mode=mode, **kwargs)
    if container_path != "":
        result = result[str(container_path)]
    return result


def access_precomputed(outer_path: Pathlike, 
                       inner_path: Pathlike, 
                       mode: str):

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
    lazy: bool = False,
    **kwargs,
) -> ArraySources:
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
        path_inner: Pathlike
        path_outer, path_inner = split_path_at_suffix(path)

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

        if lazy:
            accessor = delayed(accessor)
        if is_container:
            return accessor(path_outer, path_inner, mode=mode, **kwargs)
        else:
            return accessor(path_outer, mode=mode, **kwargs)

    elif isinstance(path, Iterable):
        return [access(p, mode, lazy, **kwargs) for p in path]
    else:
        raise ValueError("`path` must be a string or iterable of strings")


def read(path: Union[Pathlike, Iterable[str], Iterable[Path]], lazy=False, **kwargs):
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
    return access(path, mode="r", lazy=lazy, **kwargs)


def infer_coordinates_3d(arr: Arraylike, default_unit="nm") -> List[DataArray]:
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
        if isinstance(arr, da.Array) and isinstance(get_array_original(arr), zarr.core.Array):
            arr_source: zarr.core.Array = get_array_original(arr)
            coords = infer_coordinates_3d(arr_source)
        else:
            coords = [
                DataArray(np.arange(arr.shape[v]), dims=k, attrs={"units": defaultUnit})
                for k, v in _zarr_axes.items()
            ]

    return coords


def DataArrayFromFile(source_path: Pathlike):
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


def DataArrayFactory(arr: Arraylike, **kwargs):
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


def create_array(group, attrs=None, **kwargs) -> zarr.core.Array:
    name = kwargs["name"]
    overwrite = kwargs.get("overwrite", False)
    if name not in group:
        arr = group.zeros(**kwargs)
    else:
        arr = group[name]
        if not same_array_props(
            arr,
            shape=kwargs["shape"],
            dtype=kwargs["dtype"],
            compressor=kwargs["compressor"],
            chunks=kwargs["chunks"],
        ):
            arr = group.zeros(**kwargs)
        else:
            if overwrite == False:
                raise FileExistsError(
                    f"{group.path}/{name} already exists as an array. Call this function with overwrite=True to delete this array."
                )
    if attrs is not None:
        arr.attrs.put(attrs)
    return arr


def create_arrays(
    path: Union[str, Path],
    names: Sequence[str],
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str],
    compressors: Sequence[Any],
    chunks: Sequence[Sequence[int]],
    group_attrs: Dict[str, Any] = {},
    attrs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    overwrite: bool = True,
) -> Tuple[zarr.hierarchy.Group, Tuple[zarr.core.Array]]:
    """
    Use Zarr / N5 to create a collection of arrays within a group (the group will also be created, if needed). If overwrite==True,
    these arrays will be created as needed and filled with 0s. Otherwise, new arrays will be created, existing arrays with matching properties
    will be kept as-is, and existing arrays with mismatched properties will be removed and replaced with an array of 0s.      
    """

    group: zarr.hierarchy.Group = access(path, mode="a")
    group.attrs.put(group_attrs)

    if attrs is None:
        attrs = [None] * len(names)

    delayed_creator = delayed(create_array)

    arrs_delayed: List[Any] = []
    for ind in range(len(names)):
        arrs_delayed.append(
            delayed_creator(
                group=group, 
                attrs=attrs[ind], 
                name=names[ind], 
                shape=shapes[ind],
                dtype=dtypes[ind],
                compressor=compressors[ind],
                chunks=chunks[ind],
                overwrite=overwrite))

    with ProgressBar():
        arrs: Tuple[zarr.core.Array] = delayed(arrs_delayed).compute(scheduler="threads")

    return group, arrs


def access_multiscale(
    container_path: Pathlike,
    group_path: Pathlike,
    arrays: Sequence[DataArray],
    array_paths: Sequence[str], 
    array_chunks: Sequence[int],
    root_attrs: Optional[Dict[str, Any]]=None,
    group_attrs: Dict[str, Any] = {},
    compressor: Any=GZip(-1),
    attr_factory: Optional[Callable[[DataArray], Dict[str, Any]]] = None
) -> Tuple[zarr.hierarchy.group, zarr.Array]:

    root: zarr.hierarchy.Group = access(container_path, mode="a")
    if root_attrs is not None:
        root.attrs.put(root_attrs)
    
    if attr_factory is None:
        attr_factory = lambda v: {}

    shapes, dtypes, compressors, chunks, arr_attrs = [], [], [], [], []
    for p in arrays:
        shapes.append(p.shape)
        dtypes.append(p.dtype)
        compressors.append(compressor)
        chunks.append(array_chunks)
        arr_attrs.append(attr_factory(p))

    zgrp, zarrays = create_arrays(
        Path(container_path) / group_path,
        names=array_paths,
        shapes=shapes,
        dtypes=dtypes,
        compressors=compressors,
        chunks=chunks,
        group_attrs=group_attrs,
        attrs=arr_attrs,
    )

    return zgrp, zarrays


def rmtree_parallel(path: Pathlike) -> int:
    """
    Recursively remove the contents of a directory in parallel. All files are found using os.path.walk, then dask 
    is used to delete the files in parallel. Finally, the (empty) directories are removed.

    path: String, a path to the container folder, e.g. /home/user/tmp/

    return: 0

    """
    # find all the files using os.walk
    files = fwalk(path)
    if len(files) > 0:
        bg = bag.from_sequence(files)
        bg.map(lambda v: os.remove(v)).compute(scheduler="processes")
    rmtree(path)
    return 0


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


def get_array_original(arr: da.Array) -> Any:
    """
    Return the zarr array that was used to create a dask array using `da.from_array(zarr_array)`
    """
    keys = tuple(arr.dask.keys())
    return arr.dask[keys[-1]]


def fwalk(source: Pathlike, endswith: Union[str, Tuple[str, ...]] = "") -> List[str]:
    """
    Use os.walk to recursively parse a directory tree, returning a list containing the full paths
    to all files with filenames ending with `endswith`.
    """
    results = []
    for p, d, f in os.walk(source):
        for file in f:
            if file.endswith(endswith):
                results.append(os.path.join(p, file))
    return results


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