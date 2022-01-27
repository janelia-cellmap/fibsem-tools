import pytest
from fibsem_tools.io.zarr import zarr_n5_coordinate_inference
from xarray import DataArray
import numpy as np

def pixelResolutionAttr(scales, units, **kwargs):
    return {'pixelResolution': {'dimensions' : scales[::-1], 'unit': units[0]}}

def resolutionAttr(scales, **kwargs):
    return {'resolution': scales[::-1]}

def cosemAttr(scales, units, axes, translates):
    return {'transform': {'axes': axes, 'scale' : scales, 'translate': translates, 'units': units}}


@pytest.mark.parametrize('attr_factory', [pixelResolutionAttr, resolutionAttr, cosemAttr])
def test_coordinate_inference(attr_factory):
    shape = (100,200,300)
    axes = ['z', 'y', 'x']
    scales = [1.0, 2.0, 3.0]
    units = ['nm', 'nm', 'nm']
    translates = [0.0, 0.0, 0.0] 

    attr = attr_factory(scales=scales, units=units, axes=axes, translates=translates)

    result = [DataArray(translates[idx] + np.arange(shape[idx]) * scales[idx], dims=ax, attrs={"units": units[idx]}) for idx, ax in enumerate(axes)]
    coords, new_attrs = zarr_n5_coordinate_inference(shape, attr)
    for idx, r in enumerate(result):
        assert DataArray.equals(r,coords[idx])