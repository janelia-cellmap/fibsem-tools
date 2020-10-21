from fst.pyramid import lazy_pyramid
from fst.attrs import makeMultiscaleGroupAttrs
from xarray import DataArray
import dask.array as da
import numpy as np
from dataclasses import asdict


def test_multiscale_group_attrs():
    shape = (64,64,64)
    scales = (1.0, 2.0, 3.0)
    data = DataArray(da.zeros(shape), dims=('z','y','x'), coords=[np.arange(s) * scale for s, scale in zip(shape, scales)])
    [c.attrs.update({'units': 'nm'}) for c in data.coords.values()]
    pyr = lazy_pyramid(data, np.mean, (2,2,2))
    paths = [f's{idx}' for idx in range(len(pyr))]
    attrs = asdict(makeMultiscaleGroupAttrs('foo', pyr, paths))

    assert attrs == {'name': 'foo',
 'multiscales': [{'datasets': [{'path': 's0',
     'transform': {'axes': ['z', 'y', 'x'],
      'units': ['nm', 'nm', 'nm'],
      'translate': [0.0, 0.0, 0.0],
      'scale': [1.0, 2.0, 3.0]}},
    {'path': 's1',
     'transform': {'axes': ['z', 'y', 'x'],
      'units': ['nm', 'nm', 'nm'],
      'translate': [0.5, 1.0, 1.5],
      'scale': [2.0, 4.0, 6.0]}},
    {'path': 's2',
     'transform': {'axes': ['z', 'y', 'x'],
      'units': ['nm', 'nm', 'nm'],
      'translate': [1.5, 3.0, 4.5],
      'scale': [4.0, 8.0, 12.0]}},
    {'path': 's3',
     'transform': {'axes': ['z', 'y', 'x'],
      'units': ['nm', 'nm', 'nm'],
      'translate': [3.5, 7.0, 10.5],
      'scale': [8.0, 16.0, 24.0]}},
    {'path': 's4',
     'transform': {'axes': ['z', 'y', 'x'],
      'units': ['nm', 'nm', 'nm'],
      'translate': [7.5, 15.0, 22.5],
      'scale': [16.0, 32.0, 48.0]}},
    {'path': 's5',
     'transform': {'axes': ['z', 'y', 'x'],
      'units': ['nm', 'nm', 'nm'],
      'translate': [15.5, 31.0, 46.5],
      'scale': [32.0, 64.0, 96.0]}}]}],
 'axes': ('x', 'y', 'z'),
 'units': ('nm', 'nm', 'nm'),
 'scales': ((1, 1, 1),
  (2, 2, 2),
  (4, 4, 4),
  (8, 8, 8),
  (16, 16, 16),
  (32, 32, 32)),
 'pixelResolution': {'dimensions': [3.0, 2.0, 1.0], 'unit': 'nm'}}