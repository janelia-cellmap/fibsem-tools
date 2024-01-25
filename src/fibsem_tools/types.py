from typing import Any, Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class Arrayish(Protocol):
    shape: Tuple[int, ...]
    dtype: np.dtype[Any]


@runtime_checkable
class ImplicitlyChunkedArrayish(Arrayish, Protocol):
    chunks: Tuple[int, ...]


@runtime_checkable
class ExplicitlyChunkedArrayish(Arrayish, Protocol):
    chunks: Tuple[Tuple[int, ...], ...]
