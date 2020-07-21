import dask
import numpy as np
from xarray import DataArray
import dask.array as da
from fst.attrs import array_attrs, group_attrs, n5v_multiscale_group_attrs 
from fst.attrs import neuroglancer_multiscale_group_attrs, cosem_multiscale_group_attrs
from fst.io import read, access
from dataclasses import dataclass
from pathlib import Path
from fst.pyramid import lazy_pyramid
from skimage.exposure import rescale_intensity as rescale
from numcodecs import GZip
from fst.io import create_arrays
from typing import List, Tuple, Union, Dict
from dask.array.core import slices_from_chunks
from dask import delayed
from dask.utils import SerializableLock
from typing import Sequence
import os
import zarr
Pathlike = Union[str, Path]

output_chunks = (128,128,128)
output_dtype = 'uint8'
output_multiscale_format = 'neuroglancer_n5'
n5_attr_axes = {'z': 2, 'y': 1, 'x': 0}
zr_axes = {'z': 0, 'y': 1, 'x': 2}

def name_prediction_array(path: Union[str, Path]):
    from fst.io.io import split_path_at_suffix
    container, array = split_path_at_suffix(path)
    name = str(array.parts[-1])
    iterations = container.parts[-1].split('_it')[-1].split('.')[0]
    setup = container.parts[-3]
    if setup.find('setup') == -1: 
        raise ValueError(f'could not find an instance of "setup" in {setup}')
    result = f'{name}_{setup}_it{iterations}'
    return result

def save_blockwise(arr, path: Union[str, Path], block_info=None, array_location=None):
    from fst.io import access    
    import time
    import dask
    # workers create their own client object that doesn't use the same config
    #dask.config.set({'distributed.comm.timeouts.connect':'60s'})
    if array_location is not None:
        idx = array_location
    else:
        pos = block_info[0]["array-location"]    
        idx = tuple(slice(*i) for i in pos)
    num_retries = 3
    sleepdur = .5
    retval = 1
    # this idiotic construct is in here to mitigate a possible race condition when writing n5 containers via zarr
    # basically sometimes there's an OSError when writing to an array; i think this might be caused by the directory structure 
    # changing between access and write, as could happen when many workers are writing data. 
    for r in range(num_retries):
        if retval == 0: break
        try:
            sink = access(path, mode="a")
            sink[idx] = arr
            success = True
            retval = 0
        except OSError:
            time.sleep(sleepdur)
        
    return np.expand_dims(retval, tuple(range(arr.ndim))).astype(arr.dtype)
    
def prepare_multiscale_store(root_container_path: Pathlike,
                    group_path: Pathlike,
                    arrays: List,
                    array_names,
                    output_chunks: Sequence,                                        
                    root_attrs=None,
                    extra_group_attrs=None,
                    compressor = GZip(-1)) -> Tuple[zarr.hierarchy.group, zarr.Array]:

    root = access(root_container_path, mode='a')
    if root_attrs is not None:
        root.attrs.put(root_attrs)
    grp_attrs = group_attrs(arrays, axis_order='F')
    if extra_group_attrs is not None:
        grp_attrs = {**group_attrs(arrays), **extra_group_attrs}
    shapes, dtypes, compressors, chunks, arr_attrs = [],[],[],[],[]
    for ind_p, p in enumerate(arrays):            
        shapes.append(p.shape) 
        dtypes.append(p.dtype)
        compressors.append(compressor)
        chunks.append(output_chunks) 
        arr_attrs.append(array_attrs(p, axis_order='F'))
    # instead of a function with sequential arguments, what about calling a function in a loop? I find the current 
    # pattern pretty ugly
    zgrp, zarrays = create_arrays(Path(root_container_path) / group_path, 
                    names=array_names, 
                    shapes=shapes,
                    dtypes=dtypes,
                    compressors=compressors,
                    chunks=chunks,
                    group_attrs=grp_attrs,
                    array_attrs=arr_attrs)
    
    return zgrp, zarrays

def save_multiscale(arrays, zarrays) -> List[dask.delayed]:
    sources = [p.data.rechunk(za.chunks) for p,za in zip(arrays, zarrays)]
    to_store = da.store(sources, zarrays, lock=SerializableLock(), compute=False)
    return to_store
