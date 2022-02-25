import h5py
import numpy as np
from typing import Iterable, Union, Sequence
from .fibsem import FibsemDataset, FIBSEMData, FIBSEMHeader, OFFSET, MAGIC_NUMBER


def create_fibsem_h5_dataset(
    parent: Union[h5py.File, h5py.Group], name: str, data: FIBSEMData, **kwargs
):
    """
    Create a HDF5 dataset from an existing FIBSEMData object.
    """
    ds = parent.create_dataset(name, data.shape, data.dtype, **kwargs)
    for key, value in data.header.__dict__.items():
        ds.attrs[key] = value
    ds[:] = data
    return ds


def create_fibsem_h5_dataset_external(
    parent: Union[h5py.File, h5py.Group],
    name: str,
    data: FIBSEMData,
    datfile: str,
    **kwargs
):
    """
    Create a HDF5 external dataset where the data is still stored in the dat file.
    Note that the axes is rolled in the dat file versus FIBSEMData object
    """

    # FIBSEMData is usually (ChanNum, YResolution, XResolution)
    shape = (
        data.header.YResolution,
        data.header.XResolution,
        data.header.ChanNum,
    )
    parent.create_dataset(
        name, data.shape, ">i2", external=[(datfile, 1024, h5py.h5f.UNLIMITED)]
    )
    for key, value in data.header.__dict__.items():
        ds.attrs[key] = value
    ds[:] = data
    return ds


def create_fibsem_h5_file(filename: str, dataset_name: str, data: FIBSEMData, **kwargs):
    """
    Create HDF5 file with a single dataset
    """
    f = h5py.File(filename, "w")
    create_fibsem_h5_dataset(f, dataset_name, data, **kwargs)
    f.close()

def _extract_raw_header(filename: str):
    """
    Extract first kilobyte of dat file
    """
    rawfile = open(filename, "rb")
    rawbytes = rawfile.read(OFFSET)
    rawfile.close()
    assert np.frombuffer(rawbytes, '>u4', count=1)[0] == MAGIC_NUMBER
    return rawbytes

def add_raw_header_attr(
    ds: h5py.Dataset,
    filename: str
):
    """
    Extract header from filename and add it as an attribute to a HDF5 dataset
    """
    rawheader  = _extract_raw_header(filename)
    ds.attrs["RawHeader"] = np.frombuffer(rawheader, dtype='u1')

def load_fibsem_from_h5_dataset(ds: h5py.Dataset):
    """
    Create a FIBSEMData instance from a HDF5 dataset
    """
    header = FIBSEMHeader(**ds.attrs)
    if ds.shape == (header.ChanNum, header.YResolution, header.XResolution):
        # Usual order of FIBSEMData object
        data = FIBSEMData(ds[:], header)
    elif ds.shape == (header.YResolution, header.XResolution, header.ChanNum):
        # Dimensions may be in this order if using an external dataset
        data = FIBSEMData(np.rollaxis(ds[:], 2), header)
    return data
