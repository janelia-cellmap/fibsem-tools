from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import warnings
import h5py
from fibsem_tools.type import PathLike

H5_ACCESS_MODES = ("r", "r+", "w", "w-", "x", "a")

# file, group and dataset creation take both of these
H5_GROUP_KWDS = ("name", "track_order")

H5_DATASET_KWDS = (
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
    "external",
    "allow_unknown_filter",
) + H5_GROUP_KWDS

H5_FILE_KWDS = (
    "mode",
    "driver",
    "libver",
    "userblock_size",
    "swmr",
    "rdcc_nslots",
    "rdcc_nbytes",
    "rdcc_w0",
    "fs_strategy",
    "fs_persist",
    "fs_threshold",
) + H5_GROUP_KWDS


def partition_h5_kwargs(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    partition kwargs into file-creation kwargs and dataset-creation kwargs
    """
    file_kwargs = kwargs.copy()
    dataset_kwargs = {}
    for key in H5_DATASET_KWDS:
        if key in file_kwargs:
            dataset_kwargs[key] = file_kwargs.pop(key)

    return file_kwargs, dataset_kwargs


def access(
    store: PathLike, path: PathLike, mode: str, **kwargs: Any
) -> h5py.Dataset | h5py.Group:
    """
    Get or create an hdf5 dataset or group. Be advised that this function opens a file handle to the
    hdf5 file. The caller is responsible for closing that file handle, e.g.
    via access('path.h5').file.close()
    """

    # hdf5 names the root group "/", so we convert any path equal to the empty string to "/" instead
    if path == "":
        path_normalized = "/"
    else:
        path_normalized = str(path)

    if mode not in H5_ACCESS_MODES:
        msg = f"Invalid access mode. Got {mode}, expected one of {H5_ACCESS_MODES}."
        raise ValueError(msg)

    attrs = kwargs.pop("attrs", {})
    file_kwargs, dataset_kwargs = partition_h5_kwargs(**kwargs)

    h5f = h5py.File(store, mode=mode, **file_kwargs)

    if mode in ("r", "r+", "a") and (result := h5f.get(path_normalized)) is not None:
        # access a pre-existing dataset or group
        return result
    else:
        if len(dataset_kwargs) > 0:
            # create a dataset
            if "name" in dataset_kwargs:
                msg = (
                    "'Name' was provided to this function as a keyword argument. "
                    "This value will be ignored, and instead the second argument to this function "
                    "will be used as the name of the dataset or group being created."
                )
                warnings.warn(msg)

            dataset_kwargs["name"] = path_normalized
            result = h5f.create_dataset(**dataset_kwargs)
        else:
            # create a group
            result = h5f.require_group(path_normalized)

        result.attrs.update(**attrs)

        return result
