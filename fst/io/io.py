from .fibsem import read_fibsem
from pathlib import Path
from typing import Union, Iterable
import zarr


def read_n5(path: str) -> zarr.hierarchy.Group:
    result = zarr.open(zarr.N5Store(path), mode='r')
    return result


def read_zarr(path: str) -> zarr.hierarchy.Group:
    result = zarr.open(path, mode='r')
    return result


readers = dict()
readers['.dat'] = read_fibsem
readers['.n5'] = read_n5
readers['.zarr'] = read_zarr


def read(path: Union[str, Iterable[str]]):
    """

    Parameters
    ----------
    path: A path or collection of paths to image files. If `path` is a string, then the appropriate image reader will be
          selected based on the extension of the path, and the file will be read. If `path` is a collection of strings,
          it is assumed that each string is a path to an image and each will be read sequentially.

    Returns a single image, represented as an array-like object, a collection of array-like objects, or a chunked store.
    -------

    """
    if isinstance(path, str):
        return read_single(path)
    elif isinstance(path, Iterable):
        return [read_single(p) for p in path]
    else:
        raise ValueError("`path` must be an instance of string or iterable of strings")


def read_single(path: str):
    # read a single image by looking up the reader in the dict of image readers
    fmt = Path(path).suffix
    try:
        result = readers[fmt](path)
    except KeyError:
        raise ValueError(f'Cannot open images with extension {fmt}. Try one of {list(readers.keys())}')
    return result
