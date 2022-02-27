import h5py
import numpy as np
from typing import Iterable, Union, Sequence
from .fibsem import FibsemDataset, FIBSEMData, FIBSEMHeader, OFFSET, MAGIC_NUMBER, read_fibsem
import re
import os.path


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


def create_fibsem_h5_file(
    h5_filename: str, dataset_name: str, data: FIBSEMData, **kwargs
):
    """
    Create HDF5 file with a single dataset
    """
    f = h5py.File(h5_filename, "x")
    create_fibsem_h5_dataset(f, dataset_name, data, **kwargs)
    f.close()


def _extract_raw_header(filename: str):
    """
    Extract first kilobyte of dat file
    """
    rawfile = open(filename, "rb")
    rawbytes = rawfile.read(OFFSET)
    rawfile.close()
    assert np.frombuffer(rawbytes, ">u4", count=1)[0] == MAGIC_NUMBER
    return rawbytes


def add_raw_header_attr(ds: h5py.Dataset, filename: str):
    """
    Extract header from filename and add it as an attribute to a HDF5 dataset
    """
    rawheader = _extract_raw_header(filename)
    ds.attrs["RawHeader"] = np.frombuffer(rawheader, dtype="u1")


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


def create_aggregate_fibsem_h5_file(
        h5_filename: str, dataset_names: Sequence[str], data: Sequence[FIBSEMData], raw_headers: Sequence[bytes],**kwargs
):
    """
    Create aggregate FIBSEM HDF5 File
    """
    f = h5py.File(h5_filename, "x")
    for n, d, h in zip(dataset_names, data, raw_headers):
        ds = create_fibsem_h5_dataset(f, n, d, **kwargs)
        ds.attrs["RawHeader"] = np.frombuffer(h, dtype="u1")
    f.close()


def create_aggregate_fibsem_h5_files_from_dat_files(dat_files: Sequence[str], **kwargs):
    r_dat_filename = re.compile(r"(.*)_(\d+-\d+-\d+).dat")
    prefix_name = ""
    dataset_names = []
    raw_headers = []
    for dat_file in dat_files:
        dat_file_base = os.path.basename(dat_file)
        m = re.fullmatch(r_dat_filename, dat_file_base)
        dataset_names.append(m[2])
        raw_headers.append(_extract_raw_header(dat_file))
        if prefix_name == "":
            prefix_name = m[1]
        else:
            # Ensure that all dat files have the same prefix
            assert prefix_name == m[1], "All arguments must share a common prefix"
    data = read_fibsem(dat_files)
    dat_file_dir = os.path.dirname(dat_files[0])
    h5_filename = os.path.join(dat_file_dir, prefix_name + ".h5")
    create_aggregate_fibsem_h5_file(h5_filename, dataset_names, data, raw_headers, **kwargs)
