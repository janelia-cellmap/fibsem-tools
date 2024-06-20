from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pytest
from cellmap_schemas.multiscale.cosem import Group, STTransform
from xarray import DataArray
from zarr import N5FSStore

from fibsem_tools.coordinate import stt_from_array, stt_to_coords
from fibsem_tools.io.n5.core import create_dataarray
from tests.conftest import PyramidRequest


def test_sttransform() -> None:
    coords = [
        DataArray(np.arange(10), dims=("z")),
        DataArray(np.arange(10) + 5, dims=("y",), attrs={"units": "m"}),
        DataArray(10 + (np.arange(10) * 10), dims=("x",), attrs={"units": "km"}),
    ]

    data = DataArray(np.zeros((10, 10, 10)), coords=coords)
    transform = stt_from_array(data)
    assert all(
        c.equals(t) for c, t in zip(coords, stt_to_coords(transform, data.shape))
    )
    assert transform == STTransform(
        order="C",
        axes=["z", "y", "x"],
        units=["m", "m", "km"],
        translate=[0.0, 5.0, 10.0],
        scale=[1.0, 1.0, 10.0],
    )

    transform = stt_from_array(data, reverse_axes=True)
    assert transform == STTransform(
        order="F",
        axes=["x", "y", "z"],
        units=["km", "m", "m"],
        translate=[10.0, 5.0, 0.0],
        scale=[10.0, 1.0, 1.0],
    )


@pytest.mark.parametrize(
    "pyramid",
    [
        PyramidRequest(
            dims=("z", "y", "x"),
            shape=(12, 13, 14),
            scale=(1, 2, 3),
            translate=(0, 0, 0),
        ),
        PyramidRequest(
            dims=("z", "y", "x"),
            shape=(22, 53, 14),
            scale=(4, 6, 3),
            translate=(0, 0, 0),
        ),
    ],
    indirect=["pyramid"],
)
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("chunks", ["auto", (10, 10, 10)])
def test_read_array(
    pyramid: tuple[DataArray, DataArray, DataArray],
    use_dask: bool,
    chunks: Literal["auto"] | tuple[int, int, int],
    tmp_n5: str,
):
    """
    Test that we can read an n5 dataset that uses neuroglancer-compatible saalfeld lab metadata
    """
    [p.attrs.update({"foo": 10}) for p in pyramid]
    paths = ("s0", "s1", "s2")
    store = N5FSStore(tmp_n5)
    transforms = tuple(stt_from_array(array) for array in pyramid)
    group_model = Group.from_arrays(
        arrays=pyramid, chunks=(10, 10, 10), paths=paths, transforms=transforms
    )
    group = group_model.to_zarr(store=store, path="pyramid")
    if not use_dask and chunks != "auto":
        msg = f"If use_dask is False, then chunks must be 'auto'. Got {chunks} instead."
        with pytest.raises(ValueError, match=re.escape(msg)):
            observed = tuple(
                create_dataarray(element=group[path], use_dask=use_dask, chunks=chunks)
                for path in paths
            )
        return None
    else:
        observed = tuple(
            create_dataarray(element=group[path], use_dask=use_dask, chunks=chunks)
            for path in paths
        )
        result = tuple(a.equals(b) for a, b in zip(observed, pyramid))
        assert result == (True, True, True)
