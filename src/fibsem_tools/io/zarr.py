from typing import Tuple, Any
from dask import delayed
from dask.bag import from_sequence
import zarr
import os 
from .util import rmtree_parallel
from toolz import concat

def get_arrays(obj: Any) -> Tuple[zarr.core.Array]:
    result = ()
    if isinstance(obj, zarr.core.Array):
        result = (obj,)
    elif isinstance(obj, zarr.hierarchy.Group):
        if len(tuple(obj.arrays())) > 1:
            names, arrays = zip(*obj.arrays())
            result = tuple(concat(map(get_arrays, arrays)))
    return result

def delete_zbranch(branch, compute=True):
    """
    Delete a branch (group or array) from a zarr container
    """
    if isinstance(branch, zarr.hierarchy.Group):
        return delete_zgroup(branch, compute=compute)
    elif isinstance(branch, zarr.core.Array):
        return delete_zarray(branch, compute=compute)
    else:
        raise TypeError(f'The first argument to this function my be a zarr group or array, not {type(branch)}')


def delete_zgroup(zgroup, compute=True):
    """
    Delete all arrays in a zarr group
    """
    if not isinstance(zgroup, zarr.hierarchy.Group):
        raise TypeError(f'Cannot use the delete_zgroup function on object of type {type(zgroup)}')
    
    arrays = get_arrays(zgroup)
    to_delete = delayed([delete_zarray(arr, compute=False) for arr in arrays])
    
    if compute:
        return to_delete.compute()
    else:
        return to_delete


def delete_zarray(zarray, compute=True):
    """
    Delete a zarr array. 
    """

    if not isinstance(zarray, zarr.core.Array):
        raise TypeError(f'Cannot use the delete_zarray function on object of type {type(zarray)}')
    
    path = os.path.join(zarray.store.path, zarray.path)
    store = zarray.store
    branch_depth = None
    if isinstance(store, zarr.N5Store) or isinstance(store, zarr.NestedDirectoryStore):       
       branch_depth = 1
    elif isinstance(store, zarr.DirectoryStore):
        branch_depth = 0
    else:
        warnings.warn(f'Deferring to the zarr-python implementation for deleting store with type {type(store)}')
        return None
        
    result = rmtree_parallel(path, branch_depth=branch_depth, compute=compute)
    return result


def same_compressor(arr: zarr.Array, compressor) -> bool:
    """

    Determine if the compressor associated with an array is the same as a different compressor.

    arr: A zarr array
    compressor: a Numcodecs compressor, e.g. GZip(-1)
    return: True or False, depending on whether the zarr array's compressor matches the parameters (name, level) of the
    compressor.
    """
    comp = arr.compressor.compressor_config
    return comp["id"] == compressor.codec_id and comp["level"] == compressor.level


def same_array_props(
    arr: zarr.Array, shape: Tuple[int], dtype: str, compressor: Any, chunks: Tuple[int]
) -> bool:
    """

    Determine if a zarr array has properties that match the input properties.

    arr: A zarr array
    shape: A tuple. This will be compared with arr.shape.
    dtype: A numpy dtype. This will be compared with arr.dtype.
    compressor: A numcodecs compressor, e.g. GZip(-1). This will be compared with the compressor of arr.
    chunks: A tuple. This will be compared with arr.chunks
    return: True if all the properties of arr match the kwargs, False otherwise.
    """
    return (
        (arr.shape == shape)
        & (arr.dtype == dtype)
        & same_compressor(arr, compressor)
        & (arr.chunks == chunks)
    )


def zarr_array_from_dask(arr: Any) -> Any:
    """
    Return the zarr array that was used to create a dask array using `da.from_array(zarr_array)`
    """
    keys = tuple(arr.dask.keys())
    return arr.dask[keys[-1]]