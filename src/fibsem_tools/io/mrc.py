from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import dask.array as da
import mrcfile
import numpy as np
import numpy.typing as npt
from dask.array.core import normalize_chunks
from mrcfile.mrcmemmap import MrcMemmap
from xarray import DataArray

if TYPE_CHECKING:
    from collections.abc import Sequence
    from mrcfile.mrcfile import MrcFile
    from fibsem_tools.type import PathLike


def recarray_to_dict(recarray) -> dict[str, Any]:
    result = {}
    for k in recarray.dtype.names:
        if isinstance(recarray[k], np.recarray):
            result[k] = recarray_to_dict(recarray[k])
        elif hasattr(recarray, "tolist"):
            result[k] = recarray[k].tolist()
        else:
            result[k] = recarray[k]
    return result


def access(path: PathLike, mode: str, **kwargs: Any) -> MrcArrayWrapper:
    # TODO: make memory mapping optional via kwarg
    parsed = urlparse(path)
    if parsed.scheme in ("", "file"):
        return MrcArrayWrapper(MrcMemmap(parsed.path, mode=mode, **kwargs))
    else:
        msg = f"For reading .mrc files, a URL with scheme {parsed.scheme} is not valid. Scheme must be '' or 'file'."
        raise ValueError(msg)


def infer_dtype(mem: MrcFile) -> np.dtype:
    """
    Infer the datatype of an MrcMemmap array. We cannot rely on the `dtype`
    attribute because, while the MRC2014 specification does not officially support the uint8
    datatype, MRC users routinely store uint8 data as int8. This can
    only be inferred by checking if the `dmax` property of the MRC header exceeds the upper limit of
    int8 (127).
    """
    dtype = mem.data.dtype
    if (dtype == "int8") & (mem.header.dmax > 127):
        dtype = np.dtype("uint8")

    return dtype


# TODO: use the more convenient API already provided by mrcfile for this
def infer_coords(mem: MrcArrayWrapper) -> list[DataArray]:
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
        coords.append(DataArray(data=axis, dims=(key,), attrs={"units": "nm"}))

    return coords


def chunk_loader(
    fname: str, block_info: dict[Any, Any] | None = None
) -> npt.NDArray[np.int8 | np.uint8 | np.int16 | np.uint16]:
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
    return np.array(mem).astype(dtype)


def to_xarray(
    element: MrcArrayWrapper,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    use_dask: bool = True,
    coords: Any = "auto",
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataArray:
    return create_dataarray(
        element, chunks=chunks, use_dask=use_dask, coords=coords, attrs=attrs, name=name
    )


def create_dataarray(
    element: MrcArrayWrapper,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataArray:
    inferred_coords = infer_coords(element) if coords == "auto" else coords

    if name is None:
        name = Path(element.mrc._iostream.name).parts[-1]

    if attrs is None:
        attrs = recarray_to_dict(element.mrc.header)

    if use_dask:
        element = to_dask(element, chunks)

    return DataArray(element, coords=inferred_coords, attrs=attrs, name=name)


def to_dask(
    array: MrcArrayWrapper, chunks: Literal["auto"] | Sequence[int] = "auto"
) -> da.Array:
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
            if idx > 0 and (chunks[idx] != shpe) and (chunks[idx] != -1):
                msg = (
                    f"Chunk sizes of non-leading axes must match the shape of the "
                    f"array. Got chunk_size={chunks[idx]}, expected {shpe}"
                )
                raise ValueError(msg)
        _chunks = normalize_chunks(chunks, shape, dtype=dtype)

    return da.map_blocks(chunk_loader, path, chunks=_chunks, dtype=dtype)


class MrcArrayWrapper:
    """
    Wrap an mrcmemmap so that it satisfies the `ArrayLike` interface, and a few numpy-isms.
    """

    mrc: MrcMemmap
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    size: int
    flags: np.core.multiarray.flagsobj

    def __init__(self, memmap: MrcMemmap):
        self.dtype = infer_dtype(memmap)
        self.shape = memmap.data.shape
        self.size = memmap.data.size
        self.flags = memmap.data.flags
        self.mrc = memmap

    def __getitem__(self, *args: Any) -> np.ndarray:
        return self.mrc.data.__getitem__(*args).astype(self.dtype)

    def __repr__(self) -> str:
        return f"MrcArrayWrapper(shape={self.shape}, dtype={self.dtype})"
