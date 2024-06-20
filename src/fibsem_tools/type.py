from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Tuple, Union, runtime_checkable

import numpy as np


@runtime_checkable
class ArrayLike(Protocol):
    shape: Tuple[int, ...]
    dtype: np.dtype[Any]

    def __getitem__(self, *args: Any) -> ArrayLike | float: ...


@runtime_checkable
class GroupLike(Protocol):
    def values(self) -> Iterable[Union["GroupLike", ArrayLike]]:
        """
        Iterable of the children of this group
        """
        ...

    def create_group(self, name: str, **kwargs: Any) -> "GroupLike": ...

    def create_array(
        self, name: str, dtype: Any, chunks: Tuple[int, ...], compressor: Any
    ) -> ArrayLike: ...

    def __getitem__(self, *args: Any) -> ArrayLike | "GroupLike": ...


@runtime_checkable
class ImplicitlyChunkedArrayLike(ArrayLike, Protocol):
    chunks: Tuple[int, ...]


@runtime_checkable
class ExplicitlyChunkedArrayLike(ArrayLike, Protocol):
    chunks: Tuple[Tuple[int, ...], ...]


JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]
Attrs = dict[str, JSON]
PathLike = Union[Path, str]
AccessMode = Literal["w", "w-", "r", "r+", "a"]
