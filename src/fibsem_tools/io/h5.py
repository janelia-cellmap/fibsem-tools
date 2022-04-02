import h5py
from pathlib import Path
from typing import Union, Any, Dict, Tuple, Literal
import warnings

Pathlike = Union[str, Path]

H5_ACCESS_MODES = ("r", "r+", "w", "w-", "x", "a")

H5_DATASET_KWDS = ("name",
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
                     "external",
                     "allow_unknown_filter")

H5_GROUP_KWDS = ("name",
                "track_order")

H5_FILE_KWDS = ("name",
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
                  "fs_threshold")

def partition_h5_kwargs(**kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    partition kwargs into file-creation kwargs and dataset-creation kwargs
    """
    file_kwargs = kwargs.copy()
    dataset_kwargs = {}
    for key in H5_DATASET_KWDS:
        if key in file_kwargs:
            dataset_kwargs[key] = file_kwargs.pop(key)

    return file_kwargs, dataset_kwargs


def access_h5(
    store: Pathlike, path: Pathlike, mode: str, **kwargs
) -> Union[h5py.Dataset, h5py.Group]:
    """
    Docstring
    """
    if mode not in H5_ACCESS_MODES:
        raise ValueError(f"Invalid access mode. Got {mode}, expected one of {H5_ACCESS_MODES}.")

    attrs = kwargs.pop("attrs", {})
    file_kwargs, dataset_kwargs = partition_h5_kwargs(**kwargs)
    
    h5f = h5py.File(store, mode=mode, **file_kwargs)

    if mode in ("r", "r+", "a") and (result := h5f.get(path)) is not None:
        return result
    else:
        if len(dataset_kwargs) > 0:
            if 'name' in dataset_kwargs:
                warnings.warn('"Name" was provided to this function as a keyword argument. This value will be replaced with the second argument to this function.')
            dataset_kwargs["name"] = path
            result = h5f.create_dataset(**dataset_kwargs)
        else:
            result = h5f.require_group(path)

        result.attrs.update(**attrs)

        return result
