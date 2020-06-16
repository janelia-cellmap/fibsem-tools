from fst.io import read
from itertools import product
import dask.array as da
import numpy as np
from typing import Union
from mrcfile.mrcmemmap import MrcMemmap
from pathlib import Path

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
    if dtype == 'int8':
        if mem.header.dmax > 127:
            dtype = 'uint8'
    return shape, dtype


def mrc_to_dask(fname: Pathlike, stride: int=1):
    """
    Generate a dask array backed by a memory-mapped .mrc file
    """
    with read(fname) as mem:        
        shape, dtype = mrc_shape_dtype_inference(mem)        
        if mem.data.flags['C_CONTIGUOUS']:            
            concat_axis = 0
        elif mem.data.flags['F_CONTIGUOUS']:
            concat_axis = len(shape) - 1
        else:
            raise ValueError('Could not infer whether array is C or F contiguous')

    num_strides = shape[concat_axis] // stride    
    excess = shape[concat_axis] % stride
    if excess > 0:
        extra_chunk = (excess,)
    else:
        extra_chunk = ()
    distributed_chunks = (stride,) * num_strides + extra_chunk
    if concat_axis == 0:
        chunks=(distributed_chunks, *shape[1:])
    else:
        chunks=(*shape[:-1], distributed_chunks)
        
    def chunk_loader(fname, concat_axis, block_info=None):         
        idx = tuple(slice(*idcs) for idcs in block_info[None]['array-location'])
        result = np.array(read(fname).data[idx]).astype(dtype)
        return result
        
    arr = da.map_blocks(chunk_loader, fname, concat_axis, chunks=chunks, dtype=dtype)
    
    return arr