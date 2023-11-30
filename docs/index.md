# FIB-SEM tools

This is a python library used by the [cellmap](https://www.janelia.org/project-team/cellmap) project team to store and manipulate large FIB-SEM images. 

## Functionality

`fibsem-tools` provides a consistent API for reading from a variety of the formats that we routinely encounter at Cellmap.

### Basic Zarr/N5 access
```python
from fibsem_tools import read, access

# read-only access to an N5 dataset
print(read('s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16/s0'))
#> <zarr.core.Array '/em/fibsem-uint16/s0' (6368, 1600, 12000) uint16 read-only>

# read-only access to an N5 group
print(read('s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16/'))
#> <zarr.hierarchy.Group '/em/fibsem-uint16' read-only>

# read-only access to a public Zarr array
print(read('https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/plates/5966.zarr/H/16/0/0'))
#> <zarr.core.Array '/H/16/0/0' (1, 5, 1, 1080, 1080) uint16 read-only>

# open an N5 dataset for writing
print(access('/tmp/foo.n5/bar', shape=(10,10), mode='w'))
#> <zarr.core.Array '/bar' (10, 10) float64>

# open a zarr array for writing using an in-memory filesystem
x = access('memory:///tmp/foo.zarr/bar', shape=(10,10), mode='w')
print(x)
#> <zarr.core.Array '/bar' (10, 10) float64>

print(type(x.store.fs))
#> <class 'fsspec.implementations.memory.MemoryFileSystem'>
```

The same API works for non-chunked formats like `tif` and `mrc`, but without support for writing at this time.

### Coordinate-aware images

Scientific images depict measurements that were acquired in physical space. `fibsem-tools` uses [`xarray`](https://docs.xarray.dev/en/stable/index.html) to express this aspect of our imaging data.

For example, the `read_xarray` function will interpret arrays as instances of [`xarray.DataArray`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html). Correctly resolving the coordinates
requires that the array convey those coordinates in its metadata, or in the metadata of its group, in the case of hierarchical formats like Zarr / N5.

```python
from fibsem_tools import read_xarray

# read a single N5 array as a DataArray
print(read_xarray('s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16/s0'))
"""
<xarray.DataArray 's0' (z: 6368, y: 1600, x: 12000)>
dask.array<array, shape=(6368, 1600, 12000), dtype=uint16, chunksize=(384, 384, 384), chunktype=numpy.ndarray>
Coordinates:
  * z        (z) float64 0.0 5.24 10.48 15.72 ... 3.335e+04 3.336e+04 3.336e+04
  * y        (y) float64 0.0 4.0 8.0 12.0 ... 6.388e+03 6.392e+03 6.396e+03
  * x        (x) float64 0.0 4.0 8.0 12.0 ... 4.799e+04 4.799e+04 4.8e+04
Attributes:
    pixelResolution:  {'dimensions': [4.0, 4.0, 5.24], 'unit': 'nm'}
    transform:        {'axes': ['z', 'y', 'x'], 'scale': [5.24, 4.0, 4.0], 't...
"""
```

We do a lot of work with multiscale images. When the argument to `read_xarray` resolves to a group of chunked arrays that represent a multiscale pyramid, then `read_xarray` returns a [`datatree.DataTree`](https://xarray-datatree.readthedocs.io/en/latest/generated/datatree.DataTree.html#datatree.DataTree).

```python
# read a multiscale N5 group as a DataTree
print(read_xarray('s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16/'))
"""
DataTree('fibsem-uint16', parent=None)
│   Dimensions:  ()
│   Data variables:
│       *empty*
│   Attributes:
│       axes:             ['x', 'y', 'z']
│       multiscales:      [{'datasets': [{'path': 's0', 'transform': {'axes': ['z...
│       pixelResolution:  {'dimensions': [4.0, 4.0, 5.24], 'unit': 'nm'}
│       scales:           [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
│       units:            ['nm', 'nm', 'nm']
├── DataTree('s0')
│       Dimensions:  (z: 6368, y: 1600, x: 12000)
│       Coordinates:
│         * z        (z) float64 0.0 5.24 10.48 15.72 ... 3.335e+04 3.336e+04 3.336e+04
│         * y        (y) float64 0.0 4.0 8.0 12.0 ... 6.388e+03 6.392e+03 6.396e+03
│         * x        (x) float64 0.0 4.0 8.0 12.0 ... 4.799e+04 4.799e+04 4.8e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(384, 384, 384), meta=np.ndarray>
├── DataTree('s1')
│       Dimensions:  (z: 3184, y: 800, x: 6000)
│       Coordinates:
│         * z        (z) float64 2.62 13.1 23.58 34.06 ... 3.334e+04 3.335e+04 3.336e+04
│         * y        (y) float64 2.0 10.0 18.0 26.0 ... 6.378e+03 6.386e+03 6.394e+03
│         * x        (x) float64 2.0 10.0 18.0 26.0 ... 4.798e+04 4.799e+04 4.799e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(384, 384, 384), meta=np.ndarray>
├── DataTree('s2')
│       Dimensions:  (z: 1592, y: 400, x: 3000)
│       Coordinates:
│         * z        (z) float64 7.86 28.82 49.78 ... 3.331e+04 3.333e+04 3.336e+04
│         * y        (y) float64 6.0 22.0 38.0 54.0 ... 6.358e+03 6.374e+03 6.39e+03
│         * x        (x) float64 6.0 22.0 38.0 54.0 ... 4.796e+04 4.797e+04 4.799e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(192, 400, 768), meta=np.ndarray>
├── DataTree('s3')
│       Dimensions:  (z: 796, y: 200, x: 1500)
│       Coordinates:
│         * z        (z) float64 18.34 60.26 102.2 ... 3.326e+04 3.33e+04 3.334e+04
│         * y        (y) float64 14.0 46.0 78.0 110.0 ... 6.318e+03 6.35e+03 6.382e+03
│         * x        (x) float64 14.0 46.0 78.0 110.0 ... 4.792e+04 4.795e+04 4.798e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(320, 200, 960), meta=np.ndarray>
└── DataTree('s4')
        Dimensions:  (z: 398, y: 100, x: 750)
        Coordinates:
          * z        (z) float64 39.3 123.1 207.0 ... 3.316e+04 3.324e+04 3.332e+04
          * y        (y) float64 30.0 94.0 158.0 222.0 ... 6.238e+03 6.302e+03 6.366e+03
          * x        (x) float64 30.0 94.0 158.0 222.0 ... 4.784e+04 4.79e+04 4.797e+04
        Data variables:
            data     (z, y, x) uint16 dask.array<chunksize=(398, 100, 750), meta=np.ndarray>
"""

```

There are many more utilities in `fibsem-tools`. See the API documentation for a complete overview.