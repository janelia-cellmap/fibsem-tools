from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal

import tifffile

from fibsem_tools.type import ArrayLike, PathLike


def access(
    path: PathLike, mode: Literal["r"] = "r", memmap: bool = True, **kwargs: Any
) -> ArrayLike:
    if mode != "r":
        raise ValueError("Tiff files may only be accessed in read-only mode")

    if memmap:
        return tifffile.memmap(path, **kwargs)
    else:
        return tifffile.imread(path, **kwargs)
