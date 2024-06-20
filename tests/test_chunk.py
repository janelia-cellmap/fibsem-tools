from __future__ import annotations

from typing import Literal

import dask.array as da
import pytest
from xarray import DataArray

from fibsem_tools.chunk import (
    autoscale_chunk_shape,
    ensure_minimum_chunksize,
    interval_remainder,
    normalize_chunks,
    resolve_slice,
)


@pytest.mark.parametrize("chunks", ["auto", (3, 3, 3), ((3, 3, 3), (3, 3, 3))])
def test_normalize_chunks(
    chunks: Literal["auto"] | tuple[int, ...] | tuple[tuple[int, ...], ...],
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


def test_ensure_minimum_chunksize() -> None:
    data = da.zeros((10,), chunks=(2,))
    assert ensure_minimum_chunksize(data, (4,)).chunksize == (4,)

    data = da.zeros((10,), chunks=(6,))
    assert ensure_minimum_chunksize(data, (4,)).chunksize == (6,)

    data = da.zeros((10, 10), chunks=(2, 1))
    assert ensure_minimum_chunksize(data, (4, 4)).chunksize == (4, 4)

    data = da.zeros((10, 10, 10), chunks=(2, 1, 10))
    assert ensure_minimum_chunksize(data, (4, 4, 4)).chunksize == (4, 4, 10)


def test_autoscale_chunk_shape():
    chunk_shape = (1,)
    array_shape = (1000,)
    size_limit = "1KB"
    dtype = "uint8"

    assert autoscale_chunk_shape(chunk_shape, array_shape, size_limit, dtype) == (999,)

    chunk_shape = (1, 1)
    array_shape = (1000, 100)

    assert autoscale_chunk_shape(chunk_shape, array_shape, size_limit, dtype) == (
        999,
        1,
    )

    chunk_shape = (1, 1)
    array_shape = (1000, 1001)

    assert autoscale_chunk_shape(chunk_shape, array_shape, size_limit, dtype) == (
        1,
        999,
    )

    chunk_shape = (64, 64, 64)
    size_limit = "8MB"
    array_shape = (1000, 1000, 1000)

    assert autoscale_chunk_shape(chunk_shape, array_shape, size_limit, dtype) == (
        64,
        64,
        1024,
    )


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (((0, 1), (0, 1)), (0, 0)),
        (((0, 2), (0, 1)), (0, 0)),
        (((1, 2), (0, 1)), (1, 0)),
        (((0, 10), (0, 1)), (0, 0)),
    ],
)
def test_interval_remainder(data, expected):
    assert interval_remainder(*data) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            (slice(None), (0, 10)),
            slice(0, 10, 1),
        ),
        (
            (slice(0, 10, 1), (0, 10)),
            slice(0, 10, 1),
        ),
        (
            (slice(0, 10), (0, 10)),
            slice(0, 10, 1),
        ),
        (
            (slice(None), (9, 10)),
            slice(9, 10, 1),
        ),
    ],
)
def test_resolve_slice(data, expected):
    assert resolve_slice(*data) == expected
