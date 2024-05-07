import numpy as np
from xarray import DataArray
from fibsem_tools.metadata.transform import STTransform


def test_sttransform() -> None:
    coords = [
        DataArray(np.arange(10), dims=("z")),
        DataArray(np.arange(10) + 5, dims=("y",), attrs={"units": "m"}),
        DataArray(10 + (np.arange(10) * 10), dims=("x",), attrs={"units": "km"}),
    ]

    data = DataArray(np.zeros((10, 10, 10)), coords=coords)
    transform = STTransform.from_xarray(data)
    assert all(c.equals(t) for c, t in zip(coords, transform.to_coords(data.shape)))
    assert transform == STTransform(
        order="C",
        axes=["z", "y", "x"],
        units=["m", "m", "km"],
        translate=[0.0, 5.0, 10.0],
        scale=[1.0, 1.0, 10.0],
    )

    transform = STTransform.from_xarray(data, reverse_axes=True)
    assert transform == STTransform(
        order="F",
        axes=["x", "y", "z"],
        units=["km", "m", "m"],
        translate=[10.0, 5.0, 0.0],
        scale=[10.0, 1.0, 1.0],
    )
