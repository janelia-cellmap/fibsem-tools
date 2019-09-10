from .fibsem import read_fibsem
from pathlib import Path
from typing import Union, Iterable, List
import zarr
from dask import delayed
import os
import h5py

_container_extensions = ('.zarr', '.n5', '.h5')


def split_path_at_container(path):
    # check whether a path contains a valid file path to a container file, and if so which container format it is
    result = None
    pathobj = Path(path)
    if pathobj.suffix in _container_extensions:
        result = path
    else:
        for parent in pathobj.parents:
            if parent.suffix in _container_extensions:
                result = path.split(parent.suffix)
                result[0] += parent.suffix
    return result


def split_path(path: str, sep: str = ':') -> List[str]:
    """
    Split paths of the form `foo/bar.ext:baz` into `['foo/bar.ext', 'baz'] around the separator, e.g. `:`.
    If there is no separator, split `foo/bar.ext/baz` into  `['foo/bar.ext', 'baz']` using `.ext` as a separator, where
    `.ext` is a valid container extension

    Parameters
    ----------
    path: A string representing either a compound path (a/b/c:d) or a regular path (a/b/c)
    sep: A string denoting the separator to use when splitting the input string. If the separator is not found in the
    input path, a second attempt will be made to look for a supported container format extension and split the path at
    that point.

    Returns a list of strings. If the separator is not found in the input string, the second string will be empty.
    -------

    """
    parts = path.split(sep)
    result = [path, '']
    if len(parts) == 1:
        # look for a directory that ends one of the container formats
        container_split = split_path_at_container(path)
        if container_split is not None:
            result = container_split
        else:
            parts.append('')
            result = parts
    elif len(parts) == 2:
        result = parts
    elif len(parts) > 2:
        raise ValueError(f'Input string {path} contains too many instances of {sep}.')
    return result


def read_n5(dir_path: str, container_path: str = '') -> Union[zarr.hierarchy.Group, zarr.core.Array]:
    result = zarr.open(zarr.N5Store(dir_path), mode="r")
    if container_path != '':
        result = result[container_path]
    return result


def read_zarr(dir_path: str, container_path: str = '') -> Union[zarr.hierarchy.Group, zarr.core.Array]:
    result = zarr.open(dir_path, mode="r")
    if container_path != '':
        result = result[container_path]
    return result


def read_h5(dir_path: str, container_path: str = '') ->  Union[h5py._hl.files.File, h5py._hl.group.Group, h5py._hl.dataset.Dataset]:
    result = h5py.File(dir_path, mode="r")
    if container_path != '':
        result = result[container_path]
    return result


def access_n5(dir_path: str, container_path: str = '', **kwargs):
    return zarr.open(zarr.N5Store(dir_path), path=container_path, **kwargs)


readers = dict()
readers[".dat"] = read_fibsem
readers[".n5"] = read_n5
readers[".zarr"] = read_zarr
readers[".h5"] = read_h5


def access(path: Union[str, Iterable[str]], mode='r', lazy=False, **kwargs):
    """
    Enable reading and writing from array formats.

    Parameters
    ----------
    path
    mode
    lazy

    Returns an array-like object, a collection of array-like objects, a chunked store, or a dask.delayed object.
    -------

    """
    if mode == 'r':
        return read(path, lazy=lazy)
    elif mode == 'w':
        pass
    elif mode == 'a':
        pass


def read(path: Union[str, Iterable[str]], lazy=False):
    """

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
    if isinstance(path, str):
        return read_single(path, lazy)
    elif isinstance(path, Iterable):
        return [read_single(p, lazy) for p in path]
    else:
        raise ValueError("`path` must be an instance of string or iterable of strings")


def read_single(path: str, lazy=False):
    # read a single image by looking up the reader in the dict of image readers
    path_outer, path_inner = split_path(path)
    fmt = Path(path_outer).suffix
    try:
        reader = readers[fmt]
    except KeyError:
        raise ValueError(
            f"Cannot open images with extension {fmt}. Try one of {list(readers.keys())}"
        )
    if lazy:
        reader = delayed(reader)

    if path_inner == '':
        result = reader(path_outer)
    else:
        result = reader(path_outer, path_inner)

    return result


def get_umask():
    """

    Returns the current umask as an int
    -------

    """
    current_umask = os.umask(0)
    os.umask(current_umask)

    return current_umask


def chmodr(path, mode):
    """

    Parameters
    ----------
    path: A string specifying a directory to recursively process.
    mode: Either a valid `mode` argument to os.chmod, e.g. 0o777, or the string 'umask', in which case permissions are
    set based on the user's current umask value.

    Returns 0
    -------

    """

    if mode == 'umask':
        umask = get_umask()
        # convert the umask to a file permission
        mode = 0o777 - umask

    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            full_file = os.path.join(dirpath, f)
            os.chmod(full_file, mode)
    return 0
