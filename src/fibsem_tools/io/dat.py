"""
Functions for reading FIB-SEM data from Shan Xu's proprietary format. The core routines here were
adapted from David Hoffman's work which can be found in https://github.com/janelia-cellmap/FIB-SEM-Aligner/blob/master/fibsem.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from fibsem_tools.type import AccessMode, PathLike
import os
import warnings

import dask.array as da
import numpy as np
from xarray import DataArray

from fibsem_tools.coordinate import stt_coord

# This value is used to ensure that the endianness of the data is correct
MAGIC_NUMBER = 3_555_587_570
# This is the size of the header, in bytes.
OFFSET = 1024


class FIBSEMHeader:
    """
    Structure to hold header info. Note: this object is deprecated and will soon be
    removed.
    """

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __setitem__(self, *args: Any):
        return self.__dict__.__setitem__(*args)

    def __getitem__(self, *args: Any):
        return self.__dict__.__getitem__(*args)

    @property
    def keys(self):
        return self.__dict__.keys

    def update(self, **kwargs):
        """update internal dictionary"""
        self.__dict__.update(kwargs)

    def to_native_types(self) -> None:
        """
        Replace numpy numeric types with stdlib equivalents. This method modifies an
        object in place, which is terrible. Sorry about that.
        """

        for k, v in self.__dict__.items():
            if isinstance(v, np.integer):
                self.__dict__[k] = int(v)
            elif isinstance(v, np.bytes_):
                self.__dict__[k] = v.tobytes().decode("utf-8")
            elif isinstance(v, np.ndarray):
                self.__dict__[k] = v.tolist()
            elif isinstance(v, np.floating):
                self.__dict__[k] = float(v)
            else:
                self.__dict__[k] = v


class FIBSEMData(np.ndarray):
    """
    Subclass of ndarray to attach header data to fibsem data. Note: this object is
    deprecated and will soon be removed.
    """

    def __new__(cls, input_array, attrs=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.attrs = attrs
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.attrs = getattr(obj, "attrs", None)


class _DTypeDict:
    """
    Handle dtype dict manipulations. Note: this object is deprecated and will soon be
    removed.
    """

    def __init__(self, names=None, formats=None, offsets=None):
        # initialize internals to empty lists
        self.names = []
        self.formats = []
        self.offsets = []
        if names is not None:
            self.update(names, formats, offsets)

    def update(self, names, formats, offsets):
        """ """
        if isinstance(names, list):
            if len(names) == len(formats) == len(offsets):
                self.names.extend(names)
                self.formats.extend(formats)
                self.offsets.extend(offsets)
            else:
                msg = "Lengths are not equal"
                raise RuntimeError(msg)
        else:
            self.names.append(names)
            self.formats.append(formats)
            self.offsets.append(offsets)

    @property
    def dict(self):
        """Return the dict representation"""
        return {"names": self.names, "formats": self.formats, "offsets": self.offsets}

    @property
    def dtype(self):
        """return the dtype"""
        return np.dtype(self.dict)


def parse_header(header_bytes: bytes) -> FIBSEMHeader:
    """
    Parse bytes into a FIBSEMHeader.

    Parameters
    ----------

    header_bytes : bytes
        A `bytes` object representing the header of a .dat file. Conventionally,
        the header is the first 1024 bytes of the file.

    Returns
    -------

    FIBSEMHeader
    """
    header_dtype = _DTypeDict()
    header_dtype.update(
        [
            "FileMagicNum",  # Read in magic number, should be 3555587570
            "FileVersion",  # Read in file version number
            "FileType",  # Read in file type, 1 is Zeiss Neon detectors
            "SWdate",  # Read in SW date
            "TimeStep",  # Read in AI sampling time (including oversampling) in seconds
            "ChanNum",  # Read in number of channels
            "EightBit",  # Read in 8-bit data switch
        ],
        [">u4", ">u2", ">u2", ">S10", ">f8", ">u1", ">u1"],
        [0, 4, 6, 8, 24, 32, 33],
    )

    base_header = np.frombuffer(header_bytes, dtype=header_dtype.dtype, count=1)

    fibsem_header = FIBSEMHeader(**dict(zip(base_header.dtype.names, base_header[0])))
    if fibsem_header.FileMagicNum != MAGIC_NUMBER:
        msg = f"FileMagicNum should be {MAGIC_NUMBER}. Got {fibsem_header.FileMagicNum} instead."
        raise ValueError(msg)

    _scaling_offset = 36
    if fibsem_header.FileVersion == 1:
        header_dtype.update(
            "Scaling", (">f8", (fibsem_header.ChanNum, 4)), _scaling_offset
        )
    elif fibsem_header.FileVersion in {2, 3, 4, 5, 6}:
        header_dtype.update(
            "Scaling", (">f4", (fibsem_header.ChanNum, 4)), _scaling_offset
        )
    else:
        # Read in AI channel scaling factors, (col#: AI#), (row#: offset, gain,
        # 2nd order, 3rd order)
        header_dtype.update("Scaling", (">f4", (2, 4)), _scaling_offset)

    if fibsem_header.FileVersion >= 9:
        header_dtype.update(
            ["Restart", "StageMove", "FirstX", "FirstY"],
            [">u1", ">u1", ">i4", ">i4"],
            [68, 69, 70, 74],
        )

    header_dtype.update(
        ["XResolution", "YResolution"],  # X Resolution  # Y Resolution
        [">u4", ">u4"],
        [100, 104],
    )

    if fibsem_header.FileVersion in {1, 2, 3}:
        header_dtype.update(
            ["Oversampling", "AIDelay"],  # AI oversampling  # Read AI delay (
            [">u1", ">i2"],
            [108, 109],
        )
    else:
        header_dtype.update("Oversampling", ">u2", 108)  # AI oversampling

    header_dtype.update("ZeissScanSpeed", ">u1", 111)  # Scan speed (Zeiss #)
    if fibsem_header.FileVersion in {1, 2, 3}:
        header_dtype.update(
            [
                "ScanRate",  # Actual AO (scanning) rate
                "FramelineRampdownRatio",  # Frameline rampdown ratio
                "Xmin",  # X coil minimum voltage
                "Xmax",  # X coil maximum voltage
            ],
            [">f8", ">f8", ">f8", ">f8"],
            [112, 120, 128, 136],
        )
        # fibsem_header.Detmin = -10 # Detector minimum voltage
        # fibsem_header.Detmax = 10 # Detector maximum voltage
    else:
        header_dtype.update(
            [
                "ScanRate",  # Actual AO (scanning) rate
                "FramelineRampdownRatio",  # Frameline rampdown ratio
                "Xmin",  # X coil minimum voltage
                "Xmax",  # X coil maximum voltage
                "Detmin",  # Detector minimum voltage
                "Detmax",  # Detector maximum voltage
                "DecimatingFactor",  # Decimating factor
            ],
            [">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">u2"],
            [112, 116, 120, 124, 128, 132, 136],
        )

    header_dtype.update(
        [
            "AI1",  # AI Ch1
            "AI2",  # AI Ch2
            "AI3",  # AI Ch3
            "AI4",  # AI Ch4
        ],
        [">u1", ">u1", ">u1", ">u1"],
        [151, 152, 153, 154],
    )

    if fibsem_header.FileVersion >= 9:
        header_dtype.update(
            ["SampleID"],
            ["S25"],
            [155],
        )

    header_dtype.update(
        ["Notes"],
        [">S200"],
        [180],
    )

    if fibsem_header.FileVersion in {1, 2}:
        header_dtype.update(
            [
                "DetA",  # Name of detector A
                "DetB",  # Name of detector B
                "DetC",  # Name of detector C
                "DetD",  # Name of detector D
                "Mag",  # Magnification
                "PixelSize",  # Pixel size in nm
                "WD",  # Working distance in mm
                "EHT",  # EHT in kV
                "SEMApr",  # SEM aperture number
                "HighCurrent",  # high current mode (1=on, 0=off)
                "SEMCurr",  # SEM probe current in A
                "SEMRot",  # SEM scan roation in degree
                "ChamVac",  # Chamber vacuum
                "GunVac",  # E-gun vacuum
                "SEMStiX",  # SEM stigmation X
                "SEMStiY",  # SEM stigmation Y
                "SEMAlnX",  # SEM aperture alignment X
                "SEMAlnY",  # SEM aperture alignment Y
                "StageX",  # Stage position X in mm
                "StageY",  # Stage position Y in mm
                "StageZ",  # Stage position Z in mm
                "StageT",  # Stage position T in degree
                "StageR",  # Stage position R in degree
                "StageM",  # Stage position M in mm
                "BrightnessA",  # Detector A brightness (
                "ContrastA",  # Detector A contrast (
                "BrightnessB",  # Detector B brightness (
                "ContrastB",  # Detector B contrast (
                "Mode",
                # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM
                # Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External,
                # 9=External+SEM
                "FIBFocus",  # FIB focus in kV
                "FIBProb",  # FIB probe number
                "FIBCurr",  # FIB emission current
                "FIBRot",  # FIB scan rotation
                "FIBAlnX",  # FIB aperture alignment X
                "FIBAlnY",  # FIB aperture alignment Y
                "FIBStiX",  # FIB stigmation X
                "FIBStiY",  # FIB stigmation Y
                "FIBShiftX",  # FIB beam shift X in micron
                "FIBShiftY",  # FIB beam shift Y in micron
            ],
            [
                ">S10",
                ">S18",
                ">S20",
                ">S20",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">u1",
                ">u1",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">u1",
                ">f8",
                ">u1",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
                ">f8",
            ],
            [
                380,
                390,
                700,
                720,
                408,
                416,
                424,
                432,
                440,
                441,
                448,
                456,
                464,
                472,
                480,
                488,
                496,
                504,
                512,
                520,
                528,
                536,
                544,
                552,
                560,
                568,
                576,
                584,
                600,
                608,
                616,
                624,
                632,
                640,
                648,
                656,
                664,
                672,
                680,
            ],
        )
    else:
        header_dtype.update(
            [
                "DetA",  # Name of detector A
                "DetB",  # Name of detector B
                "DetC",  # Name of detector C
                "DetD",  # Name of detector D
                "Mag",  # Magnification
                "PixelSize",  # Pixel size in nm
                "WD",  # Working distance in mm
                "EHT",  # EHT in kV
                "SEMApr",  # SEM aperture number
                "HighCurrent",  # high current mode (1=on, 0=off)
                "SEMCurr",  # SEM probe current in A
                "SEMRot",  # SEM scan roation in degree
                "ChamVac",  # Chamber vacuum
                "GunVac",  # E-gun vacuum
                "SEMShiftX",  # SEM beam shift X
                "SEMShiftY",  # SEM beam shift Y
                "SEMStiX",  # SEM stigmation X
                "SEMStiY",  # SEM stigmation Y
                "SEMAlnX",  # SEM aperture alignment X
                "SEMAlnY",  # SEM aperture alignment Y
                "StageX",  # Stage position X in mm
                "StageY",  # Stage position Y in mm
                "StageZ",  # Stage position Z in mm
                "StageT",  # Stage position T in degree
                "StageR",  # Stage position R in degree
                "StageM",  # Stage position M in mm
                "BrightnessA",  # Detector A brightness (#)
                "ContrastA",  # Detector A contrast (#)
                "BrightnessB",  # Detector B brightness (#)
                "ContrastB",  # Detector B contrast (#)
                "Mode",
                # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM
                # Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External,
                # 9=External+SEM
                "FIBFocus",  # FIB focus in kV
                "FIBProb",  # FIB probe number
                "FIBCurr",  # FIB emission current
                "FIBRot",  # FIB scan rotation
                "FIBAlnX",  # FIB aperture alignment X
                "FIBAlnY",  # FIB aperture alignment Y
                "FIBStiX",  # FIB stigmation X
                "FIBStiY",  # FIB stigmation Y
                "FIBShiftX",  # FIB beam shift X in micron
                "FIBShiftY",  # FIB beam shift Y in micron
            ],
            [
                ">S10",
                ">S18",
                ">S20",
                ">S20",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">u1",
                ">u1",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">u1",
                ">f4",
                ">u1",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
            ],
            [
                380,
                390,
                410,
                430,
                460,
                464,
                468,
                472,
                480,
                481,
                490,
                494,
                498,
                502,
                510,
                514,
                518,
                522,
                526,
                530,
                534,
                538,
                542,
                546,
                550,
                554,
                560,
                564,
                568,
                572,
                600,
                604,
                608,
                620,
                624,
                628,
                632,
                636,
                640,
                644,
                648,
            ],
        )

    if fibsem_header.FileVersion >= 5:
        header_dtype.update(
            [
                "MillingXResolution",  # FIB milling X resolution
                "MillingYResolution",  # FIB milling Y resolution
                "MillingXSize",  # FIB milling X size (um)
                "MillingYSize",  # FIB milling Y size (um)
                "MillingULAng",  # FIB milling upper left inner angle (deg)
                "MillingURAng",  # FIB milling upper right inner angle (deg)
                "MillingLineTime",  # FIB line milling time (s)
                "FIBFOV",  # FIB FOV (um)
                "MillingLinesPerImage",  # FIB milling lines per image
                "MillingPIDOn",  # FIB milling PID on
                "MillingPIDMeasured",  # FIB milling PID measured (0:specimen,
                # 1:beamdump)
                "MillingPIDTarget",  # FIB milling PID target
                "MillingPIDTargetSlope",  # FIB milling PID target slope
                "MillingPIDP",  # FIB milling PID P
                "MillingPIDI",  # FIB milling PID I
                "MillingPIDD",  # FIB milling PID D
                "MachineID",  # Machine ID
                "SEMSpecimenI",  # SEM specimen current (nA)
            ],
            [
                ">u4",
                ">u4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">u2",
                ">u1",
                ">u1",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">f4",
                ">S30",
                ">f4",
            ],
            [
                652,
                656,
                660,
                664,
                668,
                672,
                676,
                680,
                684,
                686,
                689,
                690,
                694,
                698,
                702,
                706,
                800,
                980,
            ],
        )

    if fibsem_header.FileVersion >= 6:
        header_dtype.update(
            [
                "Temperature",  # Temperature (F)
                "FaradayCupI",  # Faraday cup current (nA)
                "FIBSpecimenI",  # FIB specimen current (nA)
                "BeamDump1I",  # Beam dump 1 current (nA)
                "SEMSpecimenICurrent",  # SEM specimen current (nA)
                "MillingYVoltage",  # Milling Y voltage (V)
                "FocusIndex",  # Focus index
                "FIBSliceNum",  # FIB slice #
            ],
            [">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">u4"],
            [850, 854, 858, 862, 866, 870, 874, 878],
        )
    if fibsem_header.FileVersion >= 8:
        header_dtype.update(
            [
                "BeamDump2I",  # Beam dump 2 current (nA)
                "MillingI",  # Milling current (nA)
            ],
            [">f4", ">f4"],
            [882, 886],
        )
    header_dtype.update("FileLength", ">i8", 1000)  # Read in file length in bytes

    # read header
    header = np.frombuffer(header_bytes, dtype=header_dtype.dtype, count=1)
    fibsem_header = FIBSEMHeader(**dict(zip(header.dtype.names, header[0])))
    fibsem_header.to_native_types()
    return fibsem_header


def read_header(path: PathLike) -> FIBSEMHeader:
    """
    Opens a .dat file, reads the first 1024 bytes,
    and parses those bytes as a FIBSEMHeader.

    Parameters
    ----------

    path : PathLike
        A path to a .dat file.

    Returns
    -------

    FIBSEMHeader
    """
    with open(path, mode="rb") as fobj:
        return parse_header(fobj.read(OFFSET))


def access(path: PathLike, mode: AccessMode) -> FIBSEMData:
    """

    Access a .dat file created by FIB-SEM microscopes.

    Parameters
    ----------
    path : string denoting a path to a .dat file

    Returns : an instance of FIBSEMData
    -------

    """
    if mode != "r":
        msg = (
            f"dat files can only be opened with read-only mode (r). Got {mode} instead."
        )
        raise ValueError(msg)

    # Load raw_data data file 's' or 'ieee-be.l64' Big-ian ordering, 64-bit long data
    # type
    fibsem_header = read_header(path)
    # read data
    shape = (
        fibsem_header.YResolution,
        fibsem_header.XResolution,
        fibsem_header.ChanNum,
    )

    dtype = ">u1" if fibsem_header.EightBit == 1 else ">i2"

    file_size = os.path.getsize(path)
    expected_nbytes = OFFSET + np.prod(shape) * np.dtype(dtype).itemsize

    if file_size < expected_nbytes:
        warnings.warn(
            f"The file {path} is {file_size} bytes, but a file with size of least "
            f"{expected_nbytes} bytes was expected. It will be read as an array of zeros"
        )
        raw_data = np.zeros(dtype=dtype, shape=shape)
    else:
        raw_data = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=OFFSET,
            shape=shape,
        )

    # Once read into the FIBSEMData structure it will be in memory, not memmap.
    result = FIBSEMData(raw_data, fibsem_header)
    del raw_data
    return result


def _convert_data(fibsem):
    """

    Parameters
    ----------
    fibsem

    Returns
    -------

    """
    # Convert raw_data data to electron counts
    if fibsem.header.EightBit == 1:
        scaled = np.empty(fibsem.shape, dtype=np.int16)
        detector_a, detector_b = fibsem
        if fibsem.header.AI1:
            detector_a = fibsem[0]
            scaled[0] = np.int16(
                fibsem[0]
                * fibsem.header.ScanRate
                / fibsem.header.Scaling[0, 0]
                / fibsem.header.Scaling[0, 2]
                / fibsem.header.Scaling[0, 3]
                + fibsem.header.Scaling[0, 1]
            )
            if fibsem.header.AI2:
                detector_b = fibsem[1]
                scaled[1] = np.int16(
                    fibsem[1]
                    * fibsem.header.ScanRate
                    / fibsem.header.Scaling[1, 0]
                    / fibsem.header.Scaling[1, 2]
                    / fibsem.header.Scaling[1, 3]
                    + fibsem.header.Scaling[1, 1]
                )

        elif fibsem.header.AI2:
            detector_b = fibsem[0]
            scaled[0] = np.int16(
                fibsem[0]
                * fibsem.header.ScanRate
                / fibsem.header.Scaling[0, 0]
                / fibsem.header.Scaling[0, 2]
                / fibsem.header.Scaling[0, 3]
                + fibsem.header.Scaling[0, 1]
            )

    elif fibsem.header.FileVersion in {1, 2, 3, 4, 5, 6}:
        # scaled =

        if fibsem.header.AI1:
            detector_a = (
                fibsem.header.Scaling[0, 0] + fibsem[0] * fibsem.header.Scaling[0, 1]
            )
            if fibsem.header.AI2:
                detector_b = (
                    fibsem.header.Scaling[1, 0]
                    + fibsem[1] * fibsem.header.Scaling[1, 1]
                )
                if fibsem.header.AI3:
                    (
                        fibsem.header.Scaling[2, 0]
                        + fibsem[1] * fibsem.header.Scaling[1, 1]
                    )
    else:
        pass
    return detector_a, detector_b, scaled


def chunked_fibsem_loader(
    filenames, channel_axis, pad_values=None, concat_axis=0, block_info=None
):
    """
    Load fibsem data and pad if needed. Designed to work with da.map_blocks.
    """
    idx = block_info[None]["chunk-location"]
    output_shape = block_info[None]["chunk-shape"]
    filedata = np.expand_dims(
        np.asanyarray(access(filenames[idx[0]], mode="r")), concat_axis
    )
    pad_width = np.subtract(output_shape, filedata.shape)
    if np.any(pad_width):
        if not pad_values:
            msg = "Data must be padded but no pad values were supplied!"
            raise ValueError(msg)
        padded = []
        pw = np.take(pad_width, [x for x in range(len(pad_width)) if x != channel_axis])
        for ind, channel in enumerate(np.swapaxes(filedata, 0, channel_axis)):
            padded.append(
                np.pad(
                    channel,
                    list(zip(pw * 0, pw)),
                    mode="constant",
                    constant_values=pad_values[ind],
                )
            )

        result = np.stack(padded, channel_axis)
    else:
        result = filedata
    return result


def minmax(filenames, block_info):
    idx = block_info[None]["chunk-location"][0]
    filedata = access(filenames[idx], mode="r")
    return np.expand_dims(np.array([[f.min(), f.max()] for f in filedata]), 0)


def aggregate_fibsem_metadata(fnames):
    headers = [read_header(f) for f in fnames]
    meta, shapes, dtypes = [], [], []
    # build lists of metadata for each image
    for d in headers:
        dtype = ">u1" if d.EightBit == 1 else ">i2"
        meta.append(d.__dict__)
        shapes.append((d.ChanNum, d.YResolution, d.XResolution))
        dtypes.append(dtype)
    return meta, shapes, dtypes


# TODO: make this a dataarray
class FibsemDataset:
    def __init__(self, filenames: Sequence[str]):
        """
        Create a representation of a collection of .dat files as a single dataset.
        """
        self.filenames = filenames
        self.metadata, self.shapes, self.dtypes = aggregate_fibsem_metadata(
            self.filenames
        )
        self.dims = ("c", "z", "y", "x")
        self.shape = dict(
            zip(self.axes, (len(self.filenames), *np.array(self.shapes).max(0)))
        )
        self.extrema = self.get_extrema()
        self.needs_padding = len(set(self.shapes)) > 1
        self.grid_spacing, self.coords = self.infer_coords()

    def infer_coords(self):
        lateral = self.metadata[0]["PixelSize"]
        axial = (
            abs((self.metadata[0]["WD"] - self.metadata[-1]["WD"]) / len(self.metadata))
            / 1e-6
        )
        grid_spacing = {"z": axial, "c": 1, "y": lateral, "x": lateral}
        coords = {
            k: np.arange(0, grid_spacing[k] * self.bounding_shape[k], grid_spacing[k])
            for k in grid_spacing
        }
        return grid_spacing, coords

    def get_extrema(self):
        all_extrema = da.map_blocks(
            minmax,
            self.filenames,
            chunks=((1,) * len(self.filenames), self.bounding_shape["c"], 2),
            dtype=self.dtypes[0],
        )
        result_data = da.stack([all_extrema.min(0)[:, 0], all_extrema.max(0)[:, 1]])
        return DataArray(
            result_data,
            dims=("statistic", "channel"),
            coords={"statistic": ["min", "max"], "channel": [0, 1]},
        )


def to_dask(
    data: FibsemDataset | FIBSEMData,
    chunks: Literal["auto"] | Sequence[int],
    pad_values=None,
) -> da.Array:
    if isinstance(data, FIBSEMData):
        return da.from_array(data, chunks=chunks)

    num_channels = data.bounding_shape["c"]
    if data.needs_padding:
        if pad_values is None:
            msg = "Data must be padded but no pad values were supplied!"
            raise ValueError(msg)
        elif len(pad_values) != num_channels:
            raise ValueError

    chunks = (
        (1,) * data.bounding_shape["z"],
        *[data.bounding_shape[k] for k in ("c", "y", "x")],
    )
    return da.map_blocks(
        chunked_fibsem_loader,
        data.filenames,
        data.axes["c"],
        pad_values,
        chunks=chunks,
        dtype=data.dtypes[0],
    )


def infer_coords(array: FIBSEMData) -> tuple[DataArray, ...]:
    grid_spacing_xy = array.attrs["PixelSize"]
    dims = ("x", "y", "c")
    scales = (grid_spacing_xy, grid_spacing_xy, 1)
    translates = (0, 0, 0)
    return tuple(
        stt_coord(
            length=s,
            dim=dims[idx],
            scale=scales[idx],
            translate=translates[idx],
            unit="nm",
        )
        for idx, s in enumerate(array.shape)
    )


def create_dataarray(
    element: FIBSEMData,
    *,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    name: str | None = None,
):
    if coords == "auto":
        coords = infer_coords(element)
    if use_dask:
        element = to_dask(element, chunks)

    return DataArray(element, coords, name=name)


def create_datatree(
    element: FibsemDataset,
    *,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    name: str | None = None,
):
    msg = "This behavior has not been implemented yet."
    raise NotImplementedError(msg)


def to_xarray(
    element: FIBSEMData | FibsemDataset,
    *,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    name: str | None = None,
) -> DataArray:
    """
    Convert an instance of FIBSEMData or a FibsemDataset to an xarray data structure.
    `FIBSEMData` is converted to an `xarray.DataArray`; FibsemDataset will eventually
    be converted to a `DataTree` object, but this functionality has not been implemented
    yet, so supplying a FibsemDataset to this function will raise `NotImplementedError`.

    Parameters
    ----------

    element: Union[FIBSEMData, FibsemDataset]
        The FIBSEM data to convert to an xarray data structure
    chunks: Union[Literal["auto"], Tuple[int, ...]], default is "auto"
        The chunking to use. Only meaningful if dask middleware is used.
    coords: Any, default is "auto"
        Coordinates to use. The default behavior, signalled by the string "auto",
        is to infer to coordinates from the FIBSEM data.
    attrs: Optional[Dict[str, Any]], default is `None`
        Any additional attributes to give the resulting xarray data structure. Optional.
    name: Optional[str] , default is `None`
        The name for the resulting xarray data structure. Optional.


    """
    if isinstance(element, FibsemDataset):
        return create_datatree(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            name=name,
        )
    return create_dataarray(
        element,
        chunks=chunks,
        coords=coords,
        use_dask=use_dask,
        name=name,
    )
