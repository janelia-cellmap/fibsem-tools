from fst.pyramid import downscale, prepad, lazy_pyramid, even_padding
import dask.array as da
import numpy as np

def test_even_padding():
    sizes = (10,11,12,13)
    scale = 2
    for size in sizes:
        assert (size + even_padding(size, scale)) % scale == 0

def test_prepad():
    dims = (1,2,3,4)
    for dim in dims:
        size = (10,) * dim
        chunks = (9,) * dim
        scale = (2,) * dim
        arr = da.zeros(size, chunks=chunks)
        padded = prepad(arr, scale)
        assert np.all(np.mod(padded.shape, scale) == 0)

def test_downscale_2d():
    chunks = (2,2)
    scale = (2,1)

    arr_numpy = np.array([[1,0,1,0], [0,1,0,1], [1,0,1,0], [0,1,0,1]], dtype='uint8')
    arr_dask = da.from_array(arr_numpy, chunks=chunks)
    
    downscaled_numpy_float = downscale(arr_numpy, np.mean, scale).compute()
    downscaled_dask_float = downscale(arr_dask, np.mean, scale).compute()
    
    answer_float = np.array([[.5, .5, .5, .5], [.5, .5, .5, .5]])
    assert np.array_equal(downscaled_numpy_float, answer_float)
    assert np.array_equal(downscaled_dask_float, answer_float)

    downscaled_numpy_int = downscale(arr_numpy, np.mean, scale, dtype=arr_numpy.dtype).compute()
    downscaled_dask_int = downscale(arr_dask, np.mean, scale, dtype=arr_numpy.dtype).compute()
    
    answer_int = answer_float.astype('int')
    assert np.array_equal(downscaled_numpy_int, answer_int)
    assert np.array_equal(downscaled_dask_int, answer_int)