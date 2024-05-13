from typing import Literal, Tuple, Union

import dask.array as da
import pytest
from fibsem_tools.chunk import normalize_chunks
from xarray import DataArray


@pytest.mark.parametrize("chunks", ("auto", (3, 3, 3), ((3, 3, 3), (3, 3, 3))))
def test_normalize_chunks(
    chunks: Union[Literal["auto"], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
) -> None:
    arrays = (
        DataArray(da.zeros((10, 10, 10), chunks=(4, 4, 4))),
        DataArray(da.zeros((5, 5, 5), chunks=(2, 2, 2))),
    )
    observed = normalize_chunks(arrays, chunks)
    if chunks == "auto":
        assert observed == (arrays[0].data.chunksize, arrays[1].data.chunksize)
    elif isinstance(chunks[0], int):
        assert observed == (chunks,) * len(arrays)
    else:
        assert observed == chunks
