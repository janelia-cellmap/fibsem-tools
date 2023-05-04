# FIB-SEM Tools

Tools for processing FIB-SEM data and annotations generated at Janelia Research Campus


# Installation

This package is currently distributed via pip. We are probably going to put it on conda eventually.

```bash
pip install fibsem_tools
```

# Usage

The bulk of this libary is a collection of python functions that provide a uniform interface to a variety of file + metadata formats used for storing FIB-SEM datasets. The following file formats are supported: 

| Format  | Access mode | Storage backend |
| ------------- | ------------- | ------------- |
| n5 | r/w | local, s3, gcs (via [fsspec](https://github.com/intake/filesystem_spec)) |
| zarr | r/w | local, s3, gcs (via [fsspec](https://github.com/intake/filesystem_spec)) |
| hdf5 | r | local |
| mrc | r | local |
| dat | r | local |

Because physical coordinates and metadata are extremely important for imaging data, this library uses the [`DataArray`](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html) datastructure from [`xarray`](https://github.com/pydata/xarray) to represent FIB-SEM data as arrays with spatial coordinates + metadata. E.g.,

```python
>>> from fibsem_tools import read_xarray, read
>>> from rich import print # pretty printing
>>> creds = {'anon': True} # anonymous credentials for s3
>>> group_url = 's3://janelia-cosem-datasets/jrc_sum159-1/jrc_sum159-1.n5/em/fibsem-uint16/' # path to a group of arrays on s3
>>> group = read(url, storage_options=creds) # this returns a zarr group, which in this case is a collection of arrays
>>> print(tuple(group.arrays())) # this shows all the arrays in the group
(
    ('s0', <zarr.core.Array '/em/fibsem-uint16/s0' (7632, 2800, 16000) uint16 read-only>),
    ('s1', <zarr.core.Array '/em/fibsem-uint16/s1' (3816, 1400, 8000) uint16 read-only>),
    ('s2', <zarr.core.Array '/em/fibsem-uint16/s2' (1908, 700, 4000) uint16 read-only>),
    ('s3', <zarr.core.Array '/em/fibsem-uint16/s3' (954, 350, 2000) uint16 read-only>),
    ('s4', <zarr.core.Array '/em/fibsem-uint16/s4' (477, 175, 1000) uint16 read-only>),
    ('s5', <zarr.core.Array '/em/fibsem-uint16/s5' (239, 88, 500) uint16 read-only>)
)
>>> tree = read_xarray(url, storage_options=creds) # read the group as a DataTree, a collection of xarray objects
>>> print(tree)
DataTree('fibsem-uint16', parent=None)
│   Dimensions:  ()
│   Data variables:
│       *empty*
│   Attributes:
│       axes:             ['x', 'y', 'z']
│       multiscales:      [{'datasets': [{'path': 's0', 'transform': {'axes': ['z...
│       pixelResolution:  {'dimensions': [4.0, 4.0, 4.56], 'unit': 'nm'}
│       scales:           [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 1...
│       units:            ['nm', 'nm', 'nm']
├── DataTree('s0')
│       Dimensions:  (z: 7632, y: 2800, x: 16000)
│       Coordinates:
│         * z        (z) float64 0.0 4.56 9.12 13.68 ... 3.479e+04 3.479e+04 3.48e+04
│         * y        (y) float64 0.0 4.0 8.0 12.0 ... 1.119e+04 1.119e+04 1.12e+04
│         * x        (x) float64 0.0 4.0 8.0 12.0 ... 6.399e+04 6.399e+04 6.4e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(384, 384, 384), meta=np.ndarray>
├── DataTree('s1')
│       Dimensions:  (z: 3816, y: 1400, x: 8000)
│       Coordinates:
│         * z        (z) float64 2.28 11.4 20.52 29.64 ... 3.478e+04 3.479e+04 3.48e+04
│         * y        (y) float64 2.0 10.0 18.0 26.0 ... 1.118e+04 1.119e+04 1.119e+04
│         * x        (x) float64 2.0 10.0 18.0 26.0 ... 6.398e+04 6.399e+04 6.399e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(384, 384, 384), meta=np.ndarray>
├── DataTree('s2')
│       Dimensions:  (z: 1908, y: 700, x: 4000)
│       Coordinates:
│         * z        (z) float64 6.84 25.08 43.32 ... 3.475e+04 3.477e+04 3.479e+04
│         * y        (y) float64 6.0 22.0 38.0 54.0 ... 1.116e+04 1.117e+04 1.119e+04
│         * x        (x) float64 6.0 22.0 38.0 54.0 ... 6.396e+04 6.397e+04 6.399e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(384, 384, 384), meta=np.ndarray>
├── DataTree('s3')
│       Dimensions:  (z: 954, y: 350, x: 2000)
│       Coordinates:
│         * z        (z) float64 15.96 52.44 88.92 ... 3.471e+04 3.474e+04 3.478e+04
│         * y        (y) float64 14.0 46.0 78.0 110.0 ... 1.112e+04 1.115e+04 1.118e+04
│         * x        (x) float64 14.0 46.0 78.0 110.0 ... 6.392e+04 6.395e+04 6.398e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(288, 350, 576), meta=np.ndarray>
├── DataTree('s4')
│       Dimensions:  (z: 477, y: 175, x: 1000)
│       Coordinates:
│         * z        (z) float64 34.2 107.2 180.1 ... 3.462e+04 3.469e+04 3.476e+04
│         * y        (y) float64 30.0 94.0 158.0 222.0 ... 1.104e+04 1.11e+04 1.117e+04
│         * x        (x) float64 30.0 94.0 158.0 222.0 ... 6.384e+04 6.39e+04 6.397e+04
│       Data variables:
│           data     (z, y, x) uint16 dask.array<chunksize=(384, 175, 864), meta=np.ndarray>
└── DataTree('s5')
        Dimensions:  (z: 239, y: 88, x: 500)
        Coordinates:
          * z        (z) float64 70.68 216.6 362.5 ... 3.451e+04 3.465e+04 3.48e+04
          * y        (y) float64 62.0 190.0 318.0 446.0 ... 1.094e+04 1.107e+04 1.12e+04
          * x        (x) float64 62.0 190.0 318.0 ... 6.368e+04 6.381e+04 6.393e+04
        Data variables:
            data     (z, y, x) uint16 dask.array<chunksize=(239, 88, 500), meta=np.ndarray>

>>> array = read_xarray(url + '/s0', storage_options=creds) # get one of the arrays as a dataarray
>>> print(array)
<xarray.DataArray 's0' (z: 7632, y: 2800, x: 16000)>
dask.array<s0, shape=(7632, 2800, 16000), dtype=uint16, chunksize=(384, 384, 384), chunktype=numpy.ndarray>
Coordinates:
  * z        (z) float64 0.0 4.56 9.12 13.68 ... 3.479e+04 3.479e+04 3.48e+04
  * y        (y) float64 0.0 4.0 8.0 12.0 ... 1.119e+04 1.119e+04 1.12e+04
  * x        (x) float64 0.0 4.0 8.0 12.0 ... 6.399e+04 6.399e+04 6.4e+04
Attributes:
    pixelResolution:  {'dimensions': [4.0, 4.0, 4.56], 'unit': 'nm'}
    transform:        {'axes': ['z', 'y', 'x'], 'scale': [4.56, 4.0, 4.0], 't...
```

To get the data as a numpy array (this will download *all* the chunks from s3, so be careful):
```python
>>> array = result.compute().data
```


# Development

Clone the repo: 

```bash
git clone https://github.com/janelia-cosem/fibsem-tools.git
```

Install [poetry](https://python-poetry.org/), e.g. via [pipx](https://pypa.github.io/pipx/).

Then install dependencies 
```bash
cd fibsem_tools
poetry install
```

