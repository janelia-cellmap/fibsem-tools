from .fibsem import read_fibsem
from pathlib import Path
from typing import Union, Iterable
import zarr
from dask import delayed
import os

def read_n5(path: str) -> zarr.hierarchy.Group:
    result = zarr.open(zarr.N5Store(path), mode="r")
    return result


def read_zarr(path: str) -> zarr.hierarchy.Group:
    result = zarr.open(path, mode="r")
    return result


readers = dict()
readers[".dat"] = read_fibsem
readers[".n5"] = read_n5
readers[".zarr"] = read_zarr


def read(path: Union[str, Iterable[str]], lazy=False):
    """

    Parameters
    ----------
    path: A path or collection of paths to image files. If `path` is a string, then the appropriate image reader will be
          selected based on the extension of the path, and the file will be read. If `path` is a collection of strings,
          it is assumed that each string is a path to an image and each will be read sequentially.

    lazy: A boolean, defaults to False. If True, this function returns the native file reader wrapped by
    dask.delayed. This is advantageous for distributed computing.

    Returns a single image, represented as an array-like object, a collection of array-like objects, a chunked store, or
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
    fmt = Path(path).suffix
    try:
        reader = readers[fmt]
    except KeyError:
        raise ValueError(
            f"Cannot open images with extension {fmt}. Try one of {list(readers.keys())}"
        )
    if lazy:
        reader = delayed(reader)
    result = reader(path)

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
