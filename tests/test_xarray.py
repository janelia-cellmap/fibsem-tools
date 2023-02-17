from typing import List
import numpy as np
import pytest
from xarray import DataArray

from fibsem_tools.io.zarr import zarr_n5_coordinate_inference
from pydantic_ome_ngff import Multiscale, Axis
from pydantic_ome_ngff.v05.coordinateTransformations import (
    VectorScaleTransform,
    VectorTranslationTransform,
)


def pixelResolutionAttr(scales, units, axes, translates, path):
    return {"pixelResolution": {"dimensions": scales[::-1], "unit": units[0]}}, {}


def resolutionAttr(scales, units, axes, translates, path):
    return {"resolution": scales[::-1]}, {}


def cosemAttr(scales, units, axes, translates, path):
    return {
        "transform": {
            "axes": axes,
            "scale": scales,
            "translate": translates,
            "units": units,
        }
    }, {}


def omengffAttr(
    scales: List[float],
    units: List[str],
    axes: List[str],
    translates: List[float],
    path: str,
):
    return {}, {
        "multiscales": [
            Multiscale(
                axes=[
                    Axis(name=ax, unit=u, type="space") for ax, u in zip(axes, units)
                ],
                coordinateTransformations=[
                    VectorScaleTransform(
                        scale=[
                            1,
                        ]
                        * len(scales)
                    )
                ],
                datasets=[
                    {
                        "path": path,
                        "coordinateTransformations": [
                            VectorScaleTransform(scale=scales),
                            VectorTranslationTransform(translation=translates),
                        ],
                    }
                ],
            ).dict()
        ]
    }


@pytest.mark.parametrize(
    "attr_factory", [pixelResolutionAttr, resolutionAttr, cosemAttr, omengffAttr]
)
def test_coordinate_inference(attr_factory):
    path = "foo"
    shape = (5, 6, 7)
    axes = ["z", "y", "x"]
    scales = [1.0, 2.0, 3.0]
    units = [
        "nm",
    ] * 3
    translates = [
        0.0,
    ] * 3

    array_attrs, parent_attrs = attr_factory(
        scales=scales, units=units, axes=axes, translates=translates, path=path
    )

    result = [
        DataArray(
            translates[idx] + np.arange(shape[idx]) * scales[idx],
            dims=ax,
            attrs={"unit": units[idx]},
        )
        for idx, ax in enumerate(axes)
    ]
    coords = zarr_n5_coordinate_inference(shape, array_attrs, parent_attrs, path)
    for idx, r in enumerate(result):
        assert DataArray.equals(r, coords[idx])
