import numpy as np
import xarray


def get_scale(xr: xarray.DataArray):
    return {
        dim: float(np.abs(xr.coords[dim][1] - xr.coords[dim][0])) for dim in xr.dims
    }
