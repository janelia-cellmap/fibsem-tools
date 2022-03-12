import h5py
from pathlib import Path
from typing import Union, Any, Dict
import warnings

Pathlike = Union[str, Path]


def partition_h5_kwargs(**kwargs) -> Dict[str, Any]:
    """
    partition kwargs into file-creation kwargs and dataset-creation kwargs
    """
    file_kwargs = kwargs.copy()
    dataset_kwargs = {}
    for key in ("name", "shape", "dtype", "data", "chunks", "compression", "compression_opts", "shuffle", "fletcher32", "external", "allow_unknown_filter"):
        if key in file_kwargs:
            dataset_kwargs[key] = file_kwargs.pop(key)

    if (data := kwargs.get("data")) is not None:
        if "dtype" in kwargs and (dtype := data.dtype) != kwargs["dtype"]:
            warnings.warn(
                f"{dtype=} and {data=} were provided as keyword arguments but they conflict. dtype will be inferred from `data`"
            )
        dataset_kwargs["dtype"] = data.dtype

        if "shape" in kwargs and (shape := data.shape) != kwargs["shape"]:
            warnings.warn(
                f"{shape=} and {data=} were provided as keyword arguments but they conflict. shape will be inferred from `data`"
            )

        dataset_kwargs["shape"] = data.shape

    return file_kwargs, dataset_kwargs


def access_h5(
    store: Pathlike, path: Pathlike, mode: str, **kwargs
) -> Union[h5py.Dataset, h5py.Group]:

    attrs = kwargs.pop("attrs", {})
    file_kwargs, dataset_kwargs = partition_h5_kwargs(**kwargs)
    h5f = h5py.File(store, mode=mode, **file_kwargs)

    if mode in ("r", "a") and (result := h5f.get(path)) is not None:
        return result

    if len(dataset_kwargs) > 0:
        shape = dataset_kwargs.pop("shape")
        dtype = dataset_kwargs.pop("dtype")
        result = h5f.require_dataset(path, shape, dtype, **dataset_kwargs)
    else:
        result = h5f.require_group(path)

    if mode != "r" and len(attrs) > 0:
        result.attrs.update(**attrs)

    return result
