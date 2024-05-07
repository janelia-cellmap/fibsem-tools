from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, Union, Dict, List

import fsspec
from typing import Protocol, runtime_checkable

from xarray import DataArray

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]
Attrs = Dict[str, JSON]
PathLike = Union[Path, str]
AccessMode = Literal["w", "w-", "r", "r+", "a"]


@runtime_checkable
class ArrayLike(Protocol):
    shape: Tuple[int, ...]
    dtype: Any

    def __getitem__(self, *args: Any) -> "ArrayLike" | float:
        ...


@runtime_checkable
class GroupLike(Protocol):
    def values(self) -> Iterable[Union["GroupLike", ArrayLike]]:
        """
        Iterable of the children of this group
        """
        ...

    def create_group(self, name: str, **kwargs: Any) -> "GroupLike":
        ...

    def create_array(
        self, name: str, dtype: Any, chunks: Tuple[int, ...], compressor: Any
    ) -> ArrayLike:
        ...

    def __getitem__(self, *args: Any) -> ArrayLike | "GroupLike":
        ...


def split_by_suffix(uri: PathLike, suffixes: Sequence[str]) -> Tuple[str, str, str]:
    """
    Given a string and a collection of suffixes, return
    the string split at the last instance of any element of the string
    containing one of the suffixes, as well as the suffix.
    If the last element of the string bears a suffix, return the string,
    the empty string, and the suffix.
    """
    protocol: Optional[str]
    subpath: str
    protocol, subpath = fsspec.core.split_protocol(str(uri))
    if protocol is None:
        separator = os.path.sep
    else:
        separator = "/"
    parts = Path(subpath).parts
    suffixed = [Path(part).suffix in suffixes for part in parts]

    if not any(suffixed):
        msg = f"No path elements found with the suffix(es) {suffixes} in {uri}"
        raise ValueError(msg)

    index = [idx for idx, val in enumerate(suffixed) if val][-1]
    if index == (len(parts) - 1):
        pre, post = subpath, ""
    else:
        pre, post = (
            separator.join([p.strip(separator) for p in parts[: index + 1]]),
            separator.join([p.strip(separator) for p in parts[index + 1 :]]),
        )

    suffix = Path(pre).suffix
    if protocol:
        pre = f"{protocol}://{pre}"
    return pre, post, suffix


def normalize_chunks(
    arrays: Sequence[DataArray],
    chunks: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...], Literal["auto"]],
) -> Tuple[Tuple[int, ...], ...]:
    """
    Normalize a chunk specification, given a list of arrays.

    Parameters
    ----------

    arrays: Sequence[DataArray]
        The list of arrays to define chunks for.
    chunks: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...], Literal["auto"]]
        The specification of chunks. This parameter is either a tuple of tuple of ints,
        in which case it is already normalized and it passes right through, or it is
        a tuple of ints, which will be "broadcast" to the length of `arrays`, or it is
        the string "auto", in which case the existing chunks on the arrays with be used
        if they are chunked, and otherwise chunks will be set to the shape of each
        array.

    Returns
    -------
        Tuple[Tuple[int, ...], ...]
    """
    result: Tuple[Tuple[int, ...]] = ()
    if chunks == "auto":
        for arr in arrays:
            if arr.chunks is None:
                result += (arr.shape,)
            else:
                # use the chunksize property of the underlying dask array
                result += (arr.data.chunksize,)
    elif all(isinstance(c, tuple) for c in chunks):
        result = chunks
    else:
        all_ints = all((isinstance(c, int) for c in chunks))
        if all_ints:
            result = (chunks,) * len(arrays)
        else:
            msg = f"All values in chunks must be ints. Got {chunks}"
            raise ValueError(msg)

    assert len(result) == len(arrays)
    assert tuple(map(len, result)) == tuple(
        x.ndim for x in arrays
    ), "Number of chunks per array does not equal rank of arrays"
    return result
