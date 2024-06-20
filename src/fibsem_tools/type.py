from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    import numpy as np


@runtime_checkable
class ArrayLike(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    def __getitem__(self, *args: Any) -> ArrayLike | float: ...


@runtime_checkable
class GroupLike(Protocol):
    def values(self) -> Iterable[GroupLike | ArrayLike]:
        """
        Iterable of the children of this group
        """
        ...

    def create_group(self, name: str, **kwargs: Any) -> GroupLike: ...

    def create_array(
        self, name: str, dtype: Any, chunks: tuple[int, ...], compressor: Any
    ) -> ArrayLike: ...

    def __getitem__(self, *args: Any) -> ArrayLike | GroupLike: ...


@runtime_checkable
class ImplicitlyChunkedArrayLike(ArrayLike, Protocol):
    chunks: tuple[int, ...]


@runtime_checkable
class ExplicitlyChunkedArrayLike(ArrayLike, Protocol):
    chunks: tuple[tuple[int, ...], ...]


JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
Attrs = dict[str, JSON]
PathLike = Path | str
AccessMode = Literal["w", "w-", "r", "r+", "a"]
