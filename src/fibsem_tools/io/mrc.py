from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple, Union
from urllib.parse import urlparse

import dask.array as da
import mrcfile
import numpy as np
import numpy.typing as npt
import xarray
from dask.array.core import normalize_chunks
from mrcfile.mrcfile import MrcFile
from mrcfile.mrcmemmap import MrcMemmap

from fibsem_tools.type import PathLike


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


def access(path: PathLike, mode: str, **kwargs) -> MrcArrayWrapper:
    # todo: make memory mapping optional via kwarg
    parsed = urlparse(path)
    if parsed.scheme in ("", "file"):
        return MrcArrayWrapper(MrcMemmap(parsed.path, mode=mode, **kwargs))
    else:
        msg = f"For reading .mrc files, a URL with scheme {parsed.scheme} is not valid. Scheme must be '' or 'file'."
        raise ValueError(msg)


def infer_dtype(mem: MrcFile) -> npt.DTypeLike:
    """
    Infer the datatype of an MrcMemmap array. We cannot rely on the `dtype`
    attribute because, whyile the MRC2014 specification does not officially support the uint8
    datatype, MRC users routinely store uint8 data as int8. This can
    only be inferred by checking if the header.dmax propert exceeds the upper limit of
    int8 (127).
    """
    dtype = mem.data.dtype
    if (dtype == "int8") & (mem.header.dmax > 127):
        dtype = "uint8"

    return dtype


# todo: use the more convenient API already provided by mrcfile for this
def infer_coords(mem: MrcArrayWrapper) -> List[xarray.DataArray]:
    header = mem.mrc.header
    grid_size_angstroms = header.cella
    coords = []
    # round to this many decimal places when calculting the grid spacing, in nm
    grid_spacing_decimals = 2

    if mem.flags["C_CONTIGUOUS"]:
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


def chunk_loader(
    fname: str, block_info=None
) -> npt.NDArray[Union[np.int8, np.uint8, np.int16, np.uint16]]:
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
    element: MrcArrayWrapper,
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
    element: MrcArrayWrapper,
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
        name = Path(element.mrc._iostream.name).parts[-1]

    if attrs is None:
        attrs = recarray_to_dict(element.mrc.header)

    if use_dask:
        element = to_dask(element, chunks)

    return xarray.DataArray(element, coords=inferred_coords, attrs=attrs, name=name)


def to_dask(
    array: MrcArrayWrapper, chunks: Union[Literal["auto"], Sequence[int]] = "auto"
):
    """
    Generate a dask array backed by a memory-mapped .mrc file.
    """
    shape = array.shape
    dtype = array.dtype
    path = array.mrc._iostream.name

    if chunks == "auto":
        _chunks = normalize_chunks((1, *(-1,) * (len(shape) - 1)), shape, dtype=dtype)
    else:
        # ensure that the last axes are complete
        for idx, shpe in enumerate(shape):
            if idx > 0:
                if (chunks[idx] != shpe) and (chunks[idx] != -1):
                    msg = (
                        f"Chunk sizes of non-leading axes must match the shape of the "
                        f"array. Got chunk_size={chunks[idx]}, expected {shpe}"
                    )
                    raise ValueError(msg)
        _chunks = normalize_chunks(chunks, shape, dtype=dtype)

    arr = da.map_blocks(chunk_loader, path, chunks=_chunks, dtype=dtype)
    return arr


class MrcArrayWrapper:
    """
    Wrap an mrcmemmap so that it satisfies the `ArrayLike` interface, and a few numpy-isms.
    """

    mrc: MrcMemmap
    dtype: np.dtype[Any]
    shape: Tuple[int, ...]
    size: int
    flags: np.core.multiarray.flagsobj

    def __init__(self, memmap: MrcMemmap):
        self.dtype = infer_dtype(memmap)
        self.shape = memmap.data.shape
        self.size = memmap.data.size
        self.flags = memmap.data.flags
        self.mrc = memmap

    def __getitem__(self, *args) -> np.ndarray:
        return self.mrc.data.__getitem__(*args).astype(self.dtype)

    def __repr__(self) -> str:
        return f"MrcArrayWrapper(shape={self.shape}, dtype={self.dtype})"
