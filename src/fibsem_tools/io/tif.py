from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal

    from fibsem_tools.type import ArrayLike, PathLike

import tifffile


def access(
    path: PathLike, *, mode: Literal["r"] = "r", memmap: bool = True, **kwargs: Any
) -> ArrayLike:
    """
    Use tifffile to open a tiff file.

    Parameters
    ----------
    path: str | Path
        The path to the tiff file.
    mode: Literal["r"],
        The access mode to use. Must be "r", to indicate reading. Writing tiff files is not
        supported at this time.
    memmap: bool = True
        Whether to open the tiff file via memory mapping. This can be useful for large images, but
        not all tiff files support it. If this argument is True, then `tifffile.memmap` is used to
        to open the file; otherwise, `tifffile.imread` is used.
    **kwargs: Any
        Additional keyword arguments.
    """
    if mode != "r":
        msg = "Tiff files may only be accessed in read-only mode"
        raise ValueError(msg)

    if memmap:
        return tifffile.memmap(path, **kwargs)
    else:
        return tifffile.imread(path, **kwargs)
