import numpy as np
import pytest

from fibsem_tools.coordinate import flip, stt_array, stt_coord


@pytest.mark.parametrize("length", [1, 10, 100])
@pytest.mark.parametrize("dim", ["a", "x"])
@pytest.mark.parametrize("scale", [1, 2.5, 3])
@pytest.mark.parametrize("unit", ["nm", "km"])
@pytest.mark.parametrize("translate", [0, -5, 5.5])
def test_stt_coord(length: int, dim: str, translate: float, scale: float, unit: str):
    coordvar = stt_coord(length, dim, scale, translate, unit)
    assert len(coordvar) == length
    assert coordvar.dims == (dim,)
    if len(coordvar) > 1:
        assert coordvar[1] - coordvar[0] == scale
    assert coordvar[0] == translate
    assert coordvar.units == unit


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dims": ("a",), "scales": (1,), "translates": (0,), "units": ("nm")},
        {
            "dims": ("a", "b"),
            "scales": (0, 1),
            "translates": (0, 3),
            "units": ("nm", "m"),
        },
    ],
)
def test_stt_from_array(kwargs):
    ndim = len(kwargs["dims"])
    data = np.zeros((10,) * ndim)
    xr = stt_array(data, **kwargs)
    assert xr.shape == data.shape
    assert xr.dims == kwargs["dims"]
    for idx, coordvar in enumerate(xr.coords.values()):
        assert coordvar.dims == (kwargs["dims"][idx],)
        if len(coordvar) > 1:
            assert coordvar[1] - coordvar[0] == kwargs["scales"][idx]
        assert coordvar[0] == kwargs["translates"][idx]
        assert coordvar.units == kwargs["units"][idx]


@pytest.mark.parametrize("flip_dims", [("a",), ("a", "b"), ("a", "b", "c")])
def test_flip(flip_dims):
    all_dims = ("a", "b", "c")
    ndim = len(all_dims)
    data = stt_array(
        np.random.randint(0, 255, (3,) * ndim),
        dims=all_dims,
        scales=(1,) * ndim,
        translates=(0,) * ndim,
        units=("mm",) * ndim,
    )
    test_selector = ()
    for dim in all_dims:
        if dim in flip_dims:
            test_selector += (slice(None, None, -1),)
        else:
            test_selector += (slice(None),)

    assert np.array_equal(flip(data, flip_dims).data, data.data[test_selector])
