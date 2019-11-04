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
        result = [path, '']
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

    Returns a list of 2 strings. If the separator is not found in the input string, the second string will be empty.
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


def access_fibsem(path, mode):
    if mode != 'r':
        raise ValueError('Fibsem data can only be accessed with mode = "r", i.e. read-only')
    return read_fibsem(path)


def access_n5(dir_path: str, container_path: str, mode, **kwargs):
    return zarr.open(zarr.N5Store(dir_path),
                     path=container_path,
                     mode=mode,
                     **kwargs)


def access_zarr(dir_path: str, container_path: str, mode, **kwargs):
    return zarr.open(dir_path,
                     path=container_path,
                     mode=mode,
                     **kwargs)


def access_h5(dir_path: str, container_path: str, mode, **kwargs):
    result = h5py.File(dir_path, mode=mode)
    if container_path != '':
        result = result[container_path]
    return result


accessors = dict()
accessors[".dat"] = access_fibsem
accessors[".n5"] = access_n5
accessors[".zarr"] = access_zarr
accessors[".h5"] = access_h5


def access(path: Union[str, Iterable[str]], mode, lazy=False, **kwargs):
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
    if isinstance(path, str):
        path_outer, path_inner = split_path(path)
        fmt = Path(path_outer).suffix
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


def read(path: Union[str, Iterable[str]], lazy=False, **kwargs):
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
            try:
                os.chmod(full_file, mode)
            except PermissionError:
                pass
    return 0
