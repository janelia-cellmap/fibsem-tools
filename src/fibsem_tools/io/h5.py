import h5py
from pathlib import Path
from typing import Union, Any, Dict, Tuple, Literal
import warnings

Pathlike = Union[str, Path]

H5_ACCESS_MODES = ("r", "r+", "w", "w-", "x", "a")

H5_DATASET_KWDS = (
    "name",
    "shape",
    "dtype",
    "data",
    "chunks",
    "compression",
    "compression_opts",
    "scaleoffset",
    "shuffle",
    "fletcher32",
    "maxshape",
    "fillvalue",
    "track_times",
    "track_order",
    "dcpl",
    "external",
    "allow_unknown_filter",
)

H5_GROUP_KWDS = ("name", "track_order")

H5_FILE_KWDS = (
    "name",
    "mode",
    "driver",
    "libver",
    "userblock_size",
    "swmr",
    "rdcc_nslots",
    "rdcc_nbytes",
    "rdcc_w0",
    "track_order",
    "fs_strategy",
    "fs_persist",
    "fs_threshold",
)


# Could use multiple inheritance here
class ManagedDataset(h5py.Dataset):
    """
    h5py.Dataset with context manager behavior
    """

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.file.close()


class ManagedGroup(h5py.Group):
    """
    h5py.Group with context manager behavior
    """

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.file.close()


def partition_h5_kwargs(**kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    partition kwargs into file-creation kwargs and dataset-creation kwargs
    """
    file_kwargs = kwargs.copy()
    dataset_kwargs = {}
    for key in kwargs:
        if key in H5_DATASET_KWDS:
            dataset_kwargs[key] = file_kwargs.pop(key)

    return file_kwargs, dataset_kwargs


def access_h5(
    store: Union[h5py.File, Pathlike], path: Pathlike, **kwargs
) -> Union[h5py.Dataset, h5py.Group]:
    """
    Docstring
    """
    attrs = kwargs.pop("attrs", {})
    mode = kwargs.get("mode", "r")
    file_kwargs, dataset_kwargs = partition_h5_kwargs(**kwargs)

    if isinstance(store, h5py.File):
        h5f = store
    else:
        h5f = h5py.File(store, **file_kwargs)

    if mode in ("r", "r+", "a"):
        # let h5py handle keyerrors
        result = h5f[path]
    else:
        if len(dataset_kwargs) > 0:
            if "name" in dataset_kwargs:
                warnings.warn(
                    '"Name" was provided to this function as a keyword argument. This value will be replaced with the second argument to this function.'
                )
            dataset_kwargs["name"] = path
            result = h5f.create_dataset(**dataset_kwargs)
        else:
            result = h5f.require_group(path)
        result.attrs.update(**attrs)

    if isinstance(result, h5py.Group):
        result = ManagedGroup(result.id)
    else:
        result = ManagedDataset(result.id)

    return result
