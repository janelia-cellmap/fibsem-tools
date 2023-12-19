from __future__ import annotations
from typing import Any, Dict, List, Literal, Sequence, Tuple, Union
import numpy.typing as npt
import dask.array as da
import mrcfile
import numpy as np
from dask.array.core import normalize_chunks
from mrcfile.mrcfile import MrcFile
from mrcfile.mrcmemmap import MrcMemmap
import xarray
from fibsem_tools.io.util import PathLike
from pathlib import Path


def recarray_to_dict(recarray) -> Dict[str, Any]:
    result = {}
    for k in recarray.dtype.names:
        if isinstance(recarray[k], np.recarray):
            result[k] = recarray_to_dict(recarray[k])
        else:
            if hasattr(recarray, "tolist"):
                result[k] = recarray[k].tolist()
            else:
                result[k] = recarray[k]
    return result


def access(path: PathLike, mode: str, **kwargs):
    # todo: make memory mapping optional via kwarg
    return MrcMemmap(path, mode=mode, **kwargs)


def infer_dtype(mem: MrcFile) -> npt.DTypeLike:
    """
    Infer the datatype of an MrcMemmap array. We cannot use the dtype
    attribute because the MRC2014 specification does not officially support the uint8
    datatype, but that doesn't stop people from storing uint8 data as int8. This can
    only be inferred by checking if the header.dmax propert exceeds the upper limit of
    int8 (127).
    """
    dtype = mem.data.dtype
    if dtype == "int8":
        if mem.header.dmax > 127:
            dtype = "uint8"
    return dtype


# todo: use the more convenient API already provided by mrcfile for this
def infer_coords(mem: MrcFile) -> List[xarray.DataArray]:
    header = mem.header
    grid_size_angstroms = header.cella
    coords = []
    # round to this many decimal places when calculting the grid spacing, in nm
    grid_spacing_decimals = 2

    if mem.data.flags["C_CONTIGUOUS"]:
        # we reverse the keys from (x,y,z) to (z,y,x) so the order matches
        # numpy indexing order
        keys = reversed(header.cella.dtype.fields.keys())
    else:
        keys = header.cella.dtype.fields.keys()
    for key in keys:
        grid_spacing = np.round(
            (grid_size_angstroms[key] / 10) / header[f"n{key}"], grid_spacing_decimals
        )
        axis = np.arange(header[f"n{key}start"], header[f"n{key}"]) * grid_spacing
        coords.append(xarray.DataArray(data=axis, dims=(key,), attrs={"units": "nm"}))

    return coords


def chunk_loader(fname, block_info=None):
    dtype = block_info[None]["dtype"]
    array_location = block_info[None]["array-location"]
    shape = block_info[None]["chunk-shape"]
    # mrc files are unchunked and c-contiguous, so the
    # offset will always be a product of the last N dimensions, the
    # size of the datatype, and the position along the first dimension
    offset_bytes = np.prod(shape[1:]) * np.dtype(dtype).itemsize * array_location[0][0]
    mrc = mrcfile.open(fname, header_only=True)
    offset = mrc.header.nbytes + mrc.header.nsymbt + offset_bytes
    mem = np.memmap(fname, dtype, "r", offset, shape)
    result = np.array(mem).astype(dtype)
    return result


def to_xarray(
    element: MrcFile,
    chunks: Literal["auto"] | Tuple[int, ...] = "auto",
    use_dask: bool = True,
    coords: Any = "auto",
    attrs: Dict[str, Any] | None = None,
    name: str | None = None,
):
    return create_dataarray(
        element, chunks=chunks, use_dask=use_dask, coords=coords, attrs=attrs, name=name
    )


def create_dataarray(
    element: MrcFile,
    chunks: Literal["auto"] | Tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: Dict[str, Any] | None = None,
    name: str | None = None,
) -> xarray.DataArray:
    if coords == "auto":
        inferred_coords = infer_coords(element)
    else:
        inferred_coords = coords

    if name is None:
        name = Path(element._iostream.name).parts[-1]

    if attrs is None:
        attrs = recarray_to_dict(element.header)

    if use_dask:
        element = to_dask(element, chunks)

    return xarray.DataArray(element, coords=inferred_coords, attrs=attrs, name=name)


def to_dask(array: MrcFile, chunks: Union[Literal["auto"], Sequence[int]] = "auto"):
    """
    Generate a dask array backed by a memory-mapped .mrc file.
    """
    shape = array.data.shape
    dtype = array.data.dtype
    path = array._iostream.name
    if chunks == "auto":
        _chunks = normalize_chunks((1, *(-1,) * (len(shape) - 1)), shape, dtype=dtype)
    else:
        # ensure that the last axes are complete
        for idx, shpe in enumerate(shape):
            if idx > 0:
                if (chunks[idx] != shpe) and (chunks[idx] != -1):
                    raise ValueError(
                        f"""
                        Chunk sizes of non-leading axes must match the shape of the 
                        array. Got chunk_size={chunks[idx]}, expected {shpe}
                        """
                    )
        _chunks = normalize_chunks(chunks, shape, dtype=dtype)

    arr = da.map_blocks(chunk_loader, path, chunks=_chunks, dtype=dtype)
    return arr
