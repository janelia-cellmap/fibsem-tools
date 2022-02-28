"""
Utilities for loading and saving FIBSEM Data as HDF5 fliles

High-level Interface
* create_default_h5_files
* create_zstd_h5_files
* create_gzip_h5_files
"""

import h5py
import numpy as np
from typing import Iterable, Union, Sequence, Tuple
import re
import os.path
import glob
import hdf5plugin
from .fibsem import (
    FibsemDataset,
    FIBSEMData,
    FIBSEMHeader,
    OFFSET,
    MAGIC_NUMBER,
    read_fibsem,
)

chunk_type = Union[Tuple[int, int], Tuple[int, int, int]]


def create_fibsem_h5_dataset(
    parent: Union[h5py.File, h5py.Group],
    name: str,
    data: FIBSEMData,
    chunks: chunk_type = None,
    **kwargs
):
    """
    Create a HDF5 dataset from an existing FIBSEMData object.

    :param parent: `h5py.File` or `h5py.Group` to place the dataset in
    :param name: Name of the dataset
    :param data: `FIBSEMData` instance
    :param chunks: tuple describing the chunk size (Y, X)
    :type chunks: tuple, optional
    :param **kwargs: Keyword arguments are forwarded to `parent.create_dataset`
    """

    # If chunks has length 2, add a 1 for the ChanNum dimension
    if isinstance(chunks, tuple) and len(chunks) == 2:
        if data.shape[0] == data.header.ChanNum:
            chunks = (1, *chunks)
        elif data.shape[2] == data.header.ChanNum:
            chunks = (*chunks, 1)

    ds = parent.create_dataset(name, data.shape, data.dtype, chunks=chunks, **kwargs)
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

    :param parent: `h5py.File` or `h5py.Group` to place the dataset in
    :param name: Name of the dataset
    :param data: `FIBSEMData` instance
    :param **kwargs: Keyword arguments are forwarded to `parent.create_dataset`
    """

    # FIBSEMData is usually (ChanNum, YResolution, XResolution)
    shape = (
        data.header.YResolution,
        data.header.XResolution,
        data.header.ChanNum,
    )
    parent.create_dataset(
        name,
        data.shape,
        ">i2",
        external=[(datfile, 1024, h5py.h5f.UNLIMITED)],
        **kwargs
    )
    for key, value in data.header.__dict__.items():
        ds.attrs[key] = value
    ds[:] = data
    return ds


def create_fibsem_h5_file(
    h5_filename: str, dataset_name: str, data: FIBSEMData, force: bool = False, **kwargs
):
    """
    Create HDF5 file with a single dataset

    :param h5_filename: Name of the HDF5 file to create
    :param dataset_name: Name of the dataset
    :param data: `FIBSEMData` instance
    :param force: Overrwrite existing HDF5 file if True. defaults to False.
    :type force: bool, optional
    :param **kwargs: Keyword arguments are forwarded to `parent.create_dataset`
    """
    # Use 'x' file permissions so we do not overwrite files
    mode = "w" if force else "x"
    f = h5py.File(h5_filename, mode)
    create_fibsem_h5_dataset(f, dataset_name, data, **kwargs)
    f.close()


def _extract_raw_header(dat_filename: str):
    """
    Extract first kilobyte of dat file

    :param dat_filename: Name of the .dat file from which to extract the header

    See also `fibsem.OFFSET`.
    """
    with open(dat_filename, "rb") as rawfile:
        rawbytes = rawfile.read(OFFSET)
    assert np.frombuffer(rawbytes, ">u4", count=1)[0] == MAGIC_NUMBER
    return rawbytes


def add_raw_header_attr(dataset: h5py.Dataset, filename: str):
    """
    Extract header from filename and add it as an attribute to a HDF5 dataset

    :param dataset: `h5py.Dataset` to which to add an attribute "RawHeader"
    """
    rawheader = _extract_raw_header(filename)
    dataset.attrs["RawHeader"] = np.frombuffer(rawheader, dtype="u1")


def load_fibsem_from_h5_dataset(dataset: h5py.Dataset):
    """
    Create a FIBSEMData instance from a HDF5 dataset

    :param dataset: `h5py.Dataset` to load to create a FIBSEMData instance.

    :returns: A `FIBSEMData` instance. Metadata are loaded from HDF5 attributes, not from the raw header.
    """
    header = FIBSEMHeader(**dataset.attrs)
    if dataset.shape == (header.ChanNum, header.YResolution, header.XResolution):
        # Usual order of FIBSEMData object
        data = FIBSEMData(dataset[:], header)
    elif dataset.shape == (header.YResolution, header.XResolution, header.ChanNum):
        # Dimensions may be in this order if using an external dataset
        data = FIBSEMData(np.rollaxis(dataset[:], 2), header)
    return data


def create_aggregate_fibsem_h5_file(
    h5_filename: str,
    dataset_names: Sequence[str],
    data: Sequence[FIBSEMData],
    raw_headers: Sequence[bytes] = None,
    force: bool = False,
    **kwargs
):
    """
    Create aggregate FIBSEM HDF5 File

    :param h5_filename: Filename of the HDF5 file to create.
    :param dataset_names: Names of the datasets to create
    :param data: A sequence of FIBSEMData instances.
    :param raw_headers: Raw headers in the dat file.
    :type raw_headers: bytes, optional
    :param force: Overrwrite existing HDF5 file if True. defaults to False.
    :type force: bool, optional

    :returns: h5_filename
    """
    # Use 'x' permissions so we do not overwrite files
    mode = "w" if force else "x"
    f = h5py.File(h5_filename, mode)
    if raw_headers is not None:
        for n, d, h in zip(dataset_names, data, raw_headers):
            dataset = create_fibsem_h5_dataset(f, n, d, **kwargs)
            dataset.attrs["RawHeader"] = np.frombuffer(h, dtype="u1")
    else:
        for n, d in zip(dataset_names, data):
            dataset = create_fibsem_h5_dataset(f, n, d, **kwargs)
    f.close()
    return h5_filename


def create_aggregate_fibsem_h5_file_from_dat_filenames(
    dat_filenames: Sequence[str], h5_dirname: str = None, **kwargs
):
    """
    Create aggregate FIBSEM HDF5 File from .dat filenames

    :param dat_filenames: A sequence of dat filenames
    :param h5_filename: Name of HDF5 FILE. defaults to same directory as dat_filenames[0]
    :type h5_filename: str, optional

    :returns: h5_filename
    """
    prefix_name = ""
    dataset_names = []
    raw_headers = []

    # Parse filenames of the form prefix_0_1_2.dat
    r_dat_filename = re.compile(r"(.*)_(\d+-\d+-\d+).dat")
    for dat_filename in dat_filenames:
        dat_filename_base = os.path.basename(dat_filename)
        m = re.fullmatch(r_dat_filename, dat_filename_base)
        dataset_names.append(m[2])
        raw_headers.append(_extract_raw_header(dat_filename))
        if prefix_name == "":
            prefix_name = m[1]
        else:
            # Ensure that all dat files have the same prefix
            assert prefix_name == m[1], "All arguments must share a common prefix"

    data = read_fibsem(dat_filenames)
    if h5_dirname is None:
        h5_dirname = os.path.dirname(dat_filenames[0])
    h5_filename = os.path.join(h5_dirname, prefix_name + ".h5")

    return create_aggregate_fibsem_h5_file(
        h5_filename, dataset_names, data, raw_headers, **kwargs
    )


def create_default_h5_files(source_data, **kwargs):
    """
    Create HDF5 files with default settings

    This is a high-level convenience function.

    :param source_data: Source data. .dat filenames, a directory name, or a glob
    :param chunk: Chunk size as tuple (Y, X)

    :return: list of HDF5 filenames or `None`
    """
    if os.path.isdir(source_data):
        source_data = os.path.join(source_data, "*.dat")

    if "*" in source_data:
        source_data = glob.glob(source_data)

    if not isinstance(source_data, list):
        source_data = [source_data]

    if isinstance(source_data[0], str):
        # If str is given, assume these are .dat filenames

        # Group dat filenames by prefix, and put them into prefix specific HDF5 files
        r_dat_filename = re.compile(r"(.*)_(\d+-\d+-\d+).dat")
        prefixes = [re.fullmatch(r_dat_filename, fn)[1] for fn in source_data]
        uniq_prefixes, ind_prefixes = np.unique(prefixes, return_inverse=True)
        h5_files = []
        for i, prefix in enumerate(uniq_prefixes):
            dat_filename_group = [
                d for d, ip in zip(source_data, ind_prefixes) if ip == i
            ]
            h5f = create_aggregate_fibsem_h5_file_from_dat_filenames(
                dat_filename_group, **kwargs
            )
            h5_files.append(h5f)

        return h5_files
    elif instanceof(source_data[0], FIBSEMData):
        # TODO handle more situations
        raise NotImplemented

    return None


def create_zstd_h5_files(source_data, aggression=1, **kwargs):
    """
    Create HDF5 files with ZSTD compression.

    This is a high-level convenience function.

    :param source_data: Source data. .dat filenames, directory, or glob
    :param aggression: Zstd aggression level. defaults to 1
    :type aggression: int, optional

    :return: list of HDF5 filenames or `None`
    """

    return create_default_h5_files(
        source_data,
        compression=hdf5plugin.ZSTD_ID,
        compression_opts=(aggression,),
        **kwargs
    )


def create_gzip_h5_files(source_data, aggression=1, **kwargs):
    """
    Create HDF5 files with built-in GZIP compression

    This is a high-level convenience function.

    :param source_data: Source data. .dat filenames, directory, or glob
    :param aggression: Gzip aggression level. defaults to 1
    :type aggression: int, optional

    :return: list of HDF5 filenames or `None`
    """
    return create_default_h5_files(
        source_data, compression="gzip", compression_opts=aggression, **kwargs
    )
