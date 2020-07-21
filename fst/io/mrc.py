from fst.io import read
from itertools import product
import dask.array as da
import numpy as np
from typing import Union
from mrcfile.mrcmemmap import MrcMemmap
from pathlib import Path
from dask.array.core import normalize_chunks

Pathlike = Union[str, Path]


def mrc_shape_dtype_inference(mem: MrcMemmap):
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


def mrc_to_dask(fname: Pathlike, chunks: tuple):
    """
    Generate a dask array backed by a memory-mapped .mrc file
    """
    with read(fname) as mem:
        shape, dtype = mrc_shape_dtype_inference(mem)
        
    chunks_ = normalize_chunks(chunks, shape)

    def chunk_loader(fname, block_info=None):
        idx = tuple(slice(*idcs) for idcs in block_info[None]["array-location"])
        result = np.array(read(fname).data[idx]).astype(dtype)
        return result

    arr = da.map_blocks(chunk_loader, fname, chunks=chunks_, dtype=dtype)

    return arr
