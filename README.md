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
>>> from fibsem_tools import read_xarray
>>> uri = 's3://janelia-cosem-datasets/jrc_sum159-1/jrc_sum159-1.n5/labels/gt/0003/crop26/labels/all/s0/'
# This is lazy, no data will be transferred from s3
>>> result = read_xarray(uri, storage_options={'anon' : True})
<xarray.DataArray 'array-7d8e0774f6b3933f0bfce111cbf473d0' (z: 400, y: 400, x: 400)>
dask.array<array, shape=(400, 400, 400), dtype=uint64, chunksize=(256, 256, 256), chunktype=numpy.ndarray>
Coordinates:
  * z        (z) float64 1.846e+04 1.847e+04 1.847e+04 ... 1.926e+04 1.926e+04
  * y        (y) float64 9.144e+03 9.146e+03 9.148e+03 ... 9.94e+03 9.942e+03
  * x        (x) float64 5.696e+04 5.697e+04 5.697e+04 ... 5.776e+04 5.776e+04
Attributes:
    name:       0003/Crop26/labels/all
    transform:  {'axes': ['z', 'y', 'x'], 'scale': [2.0, 2.0, 2.0], 'translat...
    urlpath:    s3://janelia-cosem-datasets/jrc_sum159-1/jrc_sum159-1.n5/labe...
```

To get the data as a numpy array (this will download all the chunks from s3):
```python
>>> result_local = result.compute().data
```


# Development

Clone the repo: 

```bash
git clone https://github.com/janelia-cosem/fibsem-tools.git
```

Dependencies are managed using [poetry](https://python-poetry.org/)

Install poetry via pip:

```bash
pip install poetry
```

or conda: 
```bash
conda install poetry -c conda-forge
```

Then install dependencies 
```bash
cd fibsem_tools
poetry install
```

