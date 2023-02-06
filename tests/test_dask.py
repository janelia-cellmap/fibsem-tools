import dask.array as da

from fibsem_tools.io.dask import ensure_minimum_chunksize, autoscale_chunk_shape


def test_ensure_minimum_chunksize():
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
