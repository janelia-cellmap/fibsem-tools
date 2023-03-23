import numpy as np
import pytest

from fibsem_tools.io.xr import stt_coord, stt_from_array


@pytest.mark.parametrize("length", (1, 10, 100))
@pytest.mark.parametrize("dim", ("a", "x"))
@pytest.mark.parametrize("scale", (1, 2.5, 3))
@pytest.mark.parametrize("unit", ("nm", "km"))
@pytest.mark.parametrize("translate", (0, -5, 5.5))
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
        dict(dims=("a",), scales=(1,), translates=(0,), units=("nm")),
        dict(dims=("a", "b"), scales=(0, 1), translates=(0, 3), units=("nm", "m")),
    ],
)
def test_stt_from_array(kwargs):
    ndim = len(kwargs["dims"])
    data = np.zeros((10,) * ndim)
    xr = stt_from_array(data, **kwargs)
    assert xr.shape == data.shape
    assert xr.dims == kwargs["dims"]
    for idx, coordvar in enumerate(xr.coords.values()):
        assert coordvar.dims == (kwargs["dims"][idx],)
        if len(coordvar) > 1:
            assert coordvar[1] - coordvar[0] == kwargs["scales"][idx]
        assert coordvar[0] == kwargs["translates"][idx]
        assert coordvar.units == kwargs["units"][idx]
