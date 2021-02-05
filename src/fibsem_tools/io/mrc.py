import dask.array as da
import numpy as np
from typing import Union, Tuple, Sequence
from mrcfile.mrcmemmap import MrcMemmap
from pathlib import Path
from dask.array.core import normalize_chunks

Pathlike = Union[str, Path]


def access_mrc(path: Pathlike, mode: str, **kwargs):
    return MrcMemmap(path, mode=mode, **kwargs)


def mrc_shape_dtype_inference(mem: MrcMemmap) -> Tuple[Tuple[int], str]:
    """
    Infer the shape and datatype of an MrcMemmap array. We cannot use the dtype attribute
    because the MRC2014 specification does not officially support the uint8 datatype,
    but that doesn't stop people from storing uint8 data as int8. This can only be inferred
    by checking if the header.dmax propert exceeds the upper limit of int8 (127).
    """
    shape = mem.data.shape
    dtype = mem.data.dtype
    if dtype == "int8":
        if mem.header.dmax > 127:
            dtype = "uint8"
    return shape, dtype


def mrc_chunk_loader(fname, block_info=None):
    idx = tuple(slice(*idcs) for idcs in block_info[None]["array-location"])
    result = np.array(access_mrc(fname, mode="r").data[idx]).astype(dtype)
    return result


def mrc_to_dask(urlpath: Pathlike, chunks: Union[str, Sequence[int]]):
    """
    Generate a dask array backed by a memory-mapped .mrc file
    """
    with access_mrc(urlpath, mode="r") as mem:
        shape, dtype = mrc_shape_dtype_inference(mem)

    if chunks=='auto':
        _chunks = (1, *(-1,) * (len(shape) -1))
            
    _chunks = normalize_chunks(chunks, shape, dtype)

    arr = da.map_blocks(mrc_chunk_loader, urlpath, chunks=_chunks, dtype=dtype)

    return arr
