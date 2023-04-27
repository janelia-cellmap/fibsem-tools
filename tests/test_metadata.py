from typing import List, Literal
import numpy as np
from xarray import DataArray
from fibsem_tools.io.xr import stt_from_array
from fibsem_tools.metadata.cosem import (
    ArrayAttrs,
    COSEMGroupMetadataV1,
    CosemGroupV1,
    MultiscaleMetaV1,
    COSEMGroupMetadataV2,
    MultiscaleMetaV2,
)
from fibsem_tools.metadata.neuroglancer import (
    NeuroglancerN5GroupMetadata,
    PixelResolution,
)
from fibsem_tools.metadata.transform import STTransform
import pytest

from fibsem_tools.tree import Array


def test_sttransform():
    coords = [
        DataArray(np.arange(10), dims=("z")),
        DataArray(np.arange(10) + 5, dims=("y",), attrs={"units": "m"}),
        DataArray(10 + (np.arange(10) * 10), dims=("x",), attrs={"units": "km"}),
    ]

    data = DataArray(np.zeros((10, 10, 10)), coords=coords)
    transform = STTransform.fromDataArray(data)
    assert all(c.equals(t) for c, t in zip(coords, transform.to_coords(data.shape)))
    assert transform == STTransform(
        order="C",
        axes=["z", "y", "x"],
        units=["m", "m", "km"],
        translate=[0.0, 5.0, 10.0],
        scale=[1.0, 1.0, 10.0],
    )

    transform = STTransform.fromDataArray(data, reverse_axes=True)
    assert transform == STTransform(
        order="F",
        axes=["x", "y", "z"],
        units=["km", "m", "m"],
        translate=[10.0, 5.0, 0.0],
        scale=[10.0, 1.0, 1.0],
    )


def test_neuroglancer_metadata():
    coords = [
        DataArray(np.arange(16) + 0.5, dims=("z"), attrs={"units": "nm"}),
        DataArray(np.arange(16) + 1 / 3, dims=("y",), attrs={"units": "m"}),
        DataArray(10 + (np.arange(16) * 100.1), dims=("x",), attrs={"units": "km"}),
    ]

    data = DataArray(np.zeros((16, 16, 16)), coords=coords)
    coarsen_kwargs = {"z": 2, "y": 2, "x": 2, "boundary": "trim"}
    multi = [data]

    for idx in range(3):
        multi.append(multi[-1].coarsen(**coarsen_kwargs).mean())

    neuroglancer_metadata = NeuroglancerN5GroupMetadata.fromDataArrays(multi)

    assert neuroglancer_metadata == NeuroglancerN5GroupMetadata(
        axes=["x", "y", "z"],
        units=["km", "m", "nm"],
        scales=[[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8]],
        pixelResolution=PixelResolution(dimensions=[100.1, 1.0, 1.0], unit="km"),
    )


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_cosem(version: Literal["v1", "v2"]):

    transform_base = {
        "axes": ["z", "y", "x"],
        "units": ["nm", "m", "km"],
        "translate": [0, -8, 10],
        "scale": [1.0, 1.0, 10.0],
    }
    shape_base = (16, 16, 16)
    data = stt_from_array(
        np.zeros(shape_base),
        dims=transform_base["axes"],
        units=transform_base["units"],
        translates=transform_base["translate"],
        scales=transform_base["scale"],
    )

    coarsen_kwargs = {"z": 2, "y": 2, "x": 2, "boundary": "trim"}
    multi: List[DataArray] = [data.coarsen(**coarsen_kwargs).mean()]
    multi.append(multi[-1].coarsen(**coarsen_kwargs).mean())
    paths = ["s0", "s1"]
    if version == "v1":

        g_meta = COSEMGroupMetadataV1.fromDataArrays(multi, paths=paths, name="data")

        assert g_meta == COSEMGroupMetadataV1(
            multiscales=[
                MultiscaleMetaV1(
                    name="data",
                    datasets=[
                        {"path": p, "transform": STTransform.fromDataArray(m)}
                        for p, m in zip(paths, multi)
                    ],
                )
            ]
        )
        CosemGroupV1(
            attrs=g_meta,
            values=[
                Array[ArrayAttrs](
                    name=paths[idx],
                    attrs=ArrayAttrs(transform=STTransform.fromDataArray(m)),
                    shape=m.shape,
                    dtype=str(m.dtype),
                )
                for idx, m in enumerate(multi)
            ],
        )

    else:
        g_meta = COSEMGroupMetadataV2.fromDataArrays(multi, paths=paths, name="data")
        assert g_meta == COSEMGroupMetadataV2(
            multiscales=[MultiscaleMetaV2(name="data", datasets=paths)]
        )
