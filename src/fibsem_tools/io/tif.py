from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal

    from fibsem_tools.type import ArrayLike, PathLike

import tifffile


def access(
    path: PathLike, mode: Literal["r"] = "r", memmap: bool = True, **kwargs: Any
) -> ArrayLike:
    if mode != "r":
        msg = "Tiff files may only be accessed in read-only mode"
        raise ValueError(msg)

    if memmap:
        return tifffile.memmap(path, **kwargs)
    else:
        return tifffile.imread(path, **kwargs)
