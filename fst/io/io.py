from .fibsem import read_fibsem
from pathlib import Path
from collections.abc import Sequence
from typing import Union, Iterable, List, Optional, Callable, Dict
import zarr
from dask import delayed
import os
import h5py
from shutil import rmtree
from glob import glob
from itertools import groupby
from collections import defaultdict
from dask.diagnostics import ProgressBar
_formats = ('.dat', )
_container_extensions = ('.zarr', '.n5', '.h5')
_suffixes = (*_formats, *_container_extensions)

def broadcast_kwargs(**kwargs):
    """
    For each keyword: arg in kwargs, assert that there are only 2 types of args: sequences with length = 1 
    or sequences with some length = k. Every arg with length 1 will be repeated k times, such that the return value 
    is a dict of kwargs with minimum length = k.
    """
    grouped = defaultdict(list)
    sorter = lambda v: len(v[1])
    s = sorted(kwargs.items(), key=sorter)
    for l,v in groupby(s, key=sorter):
        grouped[l].extend(v)

    assert len(grouped.keys()) <= 2
    if len(grouped.keys()) == 2:
        assert min(grouped.keys()) == 1
        output_length = max(grouped.keys())
        singletons, nonsingletons = tuple(grouped.values())
        singletons = ((k,  v * output_length) for k, v in singletons)
        result = {**dict(singletons), **dict(nonsingletons)}
    else:
        result = kwargs
    
    return result


def split_path_at_suffix(upper_path: Union[str, Path], lower_path: Union[str, Path] = '', suffixes: tuple = _suffixes) -> List[Path]:
    """
    Recursively climb a path, checking at each level of the path whether the tail of the path represents a directory
    with a container extension. Returns the path broken at the level where a container is found.  
    """
    upper, lower = Path(upper_path), Path(lower_path)
    
    if upper.suffix in suffixes:
        result = [upper, lower]
    else:
        if len(upper.parts) >= 2:
            result = split_path_at_suffix(Path(*upper.parts[:-1]), Path(upper.parts[-1], lower), suffixes)
        else:
            raise ValueError(f'Could not find any suffixes matching {suffixes} in {upper / lower}')
    
    return result


def access_fibsem(path: Union[str, Path, Iterable[str], Iterable[Path]], mode: str):
    if mode != 'r':
        raise ValueError(f'.dat files can only be accessed in read-only mode, not {mode}.')
    return read_fibsem(path)


def access_n5(dir_path: Union[str, Path], container_path: Union[str, Path], **kwargs):
    return zarr.open(zarr.N5Store(dir_path),
                     path=container_path,
                     **kwargs)


def access_zarr(dir_path: Union[str, Path], container_path: Union[str, Path], **kwargs):
    return zarr.open(dir_path,
                     path=container_path,
                     **kwargs)


def access_h5(dir_path: Union[str, Path], container_path: Union[str, Path], mode: str, **kwargs):
    result = h5py.File(dir_path, mode=mode, **kwargs)
    if container_path != '':
        result = result[container_path]
    return result


accessors: Dict[str, Callable] = {}
accessors[".dat"] = access_fibsem
accessors[".n5"] = access_n5
accessors[".zarr"] = access_zarr
accessors[".h5"] = access_h5


def access(path: Union[str, Path, Iterable[str], Iterable[Path]], mode: str, lazy: bool = False, **kwargs):
    """

    Access data on disk from a variety of array storage formats.

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
        path_inner: Union[str, Path]
        path_outer, path_inner = split_path_at_suffix(path)
        
        # str(Path('')) => '.', which we don't want for an empty trailing path
        if str(path_inner) == '.':
            path_inner = ''
        
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


def read(path: Union[str, Path, Iterable[str], Iterable[Path]], lazy=False, **kwargs):
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
    return access(path, mode='r', lazy=lazy, **kwargs)


def create_array(group, attrs, **kwargs):
    name = kwargs['name']
    overwrite = kwargs.get('overwrite', False)
    if name not in group:
        arr = group.zeros(**kwargs)
    else:
        arr = group[name]
        if (not same_array_props(arr, 
                                shape=kwargs['shape'], 
                                dtype=kwargs['dtype'], 
                                compressor=kwargs['compressor'], 
                                chunks=kwargs['chunks'])):                    
                arr = group.zeros(**kwargs)
        else:
            if overwrite == False:
                raise FileExistsError(f'{group.path}/{name} already exists as an array. Call this function with overwrite=True to delete this array.')
    arr.attrs.put(attrs)


def create_arrays(
    path: Union[str, Path],
    names: Sequence,
    shapes: Sequence,
    dtypes: Sequence,
    compressors: Sequence,
    chunks: Sequence,
    group_attrs: dict,
    array_attrs: Sequence,
    overwrite: bool=True,
    parallel: bool = True):
    """
    Use Zarr / N5 to create a collection of arrays within a group (the group will also be created, if needed). If overwrite==True,
    these arrays will be created as needed and filled with 0s. Otherwise, new arrays will be created, existing arrays with matching properties
    will be kept as-is, and existing arrays with mismatched properties will be removed and replaced with an array of 0s.      
    """

    #todo: check that all sequential arguments are the same length    
    group = access(path, mode="a")
    group.attrs.put(group_attrs)
    
    argdicts = tuple({k: v for k,v in zip(('name','shape','dtype','compressor','chunks'), vals)} for vals in zip(names, shapes, dtypes, compressors, chunks))
    
    if parallel:
        arrs = [delayed(create_array)(group, array_attrs[ind], **argdict, overwrite=overwrite) for ind, argdict in enumerate(argdicts)]
        with ProgressBar():
            delayed(arrs).compute(scheduler='threads')
    else:
        [create_array(group, array_attrs[ind], **argdict, overwrite=overwrite) for ind, argdict in enumerate(argdicts)]
    
    return group


def get_array_paths(root_path):
    if root_path[-1] != os.path.sep:
        root_path += os.path.sep
    root = read(root_path)
    if isinstance(root, zarr.hierarchy.array):
        arrays = [root]
    else:
        arrays = get_arrays(root)

    result = [g for r in arrays for g in glob(root_path + r.path + '/*')]

    return result


def get_arrays(g):
    result = []
    groups, arrays = list(g.groups()), list(g.arrays())

    if len(arrays) >= 1:
        [result.append(a[1]) for a in arrays]

    if len(groups) >= 1:
        [result.extend(get_arrays(g[1])) for g in groups]

    return result


def dask_delete(path):
    if os.path.isdir(path):
        return delayed(rmtree)(path)
    else:
        return delayed(os.unlink)(path)


def rmtree_parallel(path):
    """
    Use dask to remove the contents of a directory in parallel. Parallelization is performed over the elements in the
    directory, so this will achieve no speedup if the directory contains a single element.

    path: String, a path to the container folder, e.g. /home/user/tmp/

    return: 0

    """
    stuff = tuple(Path(path).glob('*'))
    if len(stuff) >= 1:
        _ = delayed(map(dask_delete, stuff)).compute(scheduler='threads')
    rmtree(path)
    return 0


def same_compressor(arr, compressor):
    """

    Determine if the compressor associated with an array is the same as a different compressor.

    arr: A zarr array
    compressor: a Numcodecs compressor, e.g. GZip(-1)
    return: True or False, depending on whether the zarr array's compressor matches the parameters (name, level) of the
    compressor.
    """
    comp = arr.compressor.compressor_config
    return comp['id'] == compressor.codec_id and comp['level'] == compressor.level


def same_array_props(arr, shape, dtype, compressor, chunks):
    """

    Determine if a zarr array has properties that match the input properties.

    arr: A zarr array
    shape: A tuple. This will be compared with arr.shape.
    dtype: A numpy dtype. This will be compared with arr.dtype.
    compressor: A numcodecs compressor, e.g. GZip(-1). This will be compared with the compressor of arr.
    chunks: A tuple. This will be compared with arr.chunks
    return: True if all the properties of arr match the kwargs, False otherwise.
    """
    return (arr.shape == shape) & (arr.dtype == dtype) & same_compressor(arr, compressor) & (arr.chunks == chunks)
