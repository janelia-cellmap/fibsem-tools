from typing import Any, Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class ArrayLike(Protocol):
    shape: Tuple[int, ...]
    dtype: np.dtype[Any]
