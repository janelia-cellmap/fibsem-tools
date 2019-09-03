"""
Functions for reading FIB-SEM data from Harald Hess' proprietary format
Adapted from https://github.com/janelia-cosem/FIB-SEM-Aligner/blob/master/fibsem.py
Copyright (c) 2017, David Hoffman, Davis Bennett
"""

import os
import numpy as np
from typing import Iterable, Union

# This value is used to ensure that the endianness of the data is correct
MAGIC_NUMBER = 3555587570


class FIBSEMHeader(object):
    """Structure to hold header info"""

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def update(self, **kwargs):
        """update internal dictionary"""
        self.__dict__.update(kwargs)

    def to_native_types(self):
        for k, v in self.__dict__.items():
            if isinstance(v, np.integer):
                self.__dict__[k] = int(v)
            elif isinstance(v, np.bytes_):
                self.__dict__[k] = v.tostring().decode("utf-8")
            elif isinstance(v, np.ndarray):
                self.__dict__[k] = v.tolist()
            elif isinstance(v, np.floating):
                self.__dict__[k] = float(v)
            else:
                self.__dict__[k] = v


class FIBSEMData(np.ndarray):
    """Subclass of ndarray to attach header data to fibsem data"""

    def __new__(cls, input_array, header=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.header = header
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.header = getattr(obj, "header", None)


class _DTypeDict(object):
    """Handle dtype dict manipulations"""

    def __init__(self, names=None, formats=None, offsets=None):
        # initialize internals to empty lists
        self.names = []
        self.formats = []
        self.offsets = []
        if names is not None:
            self.update(names, formats, offsets)

    def update(self, names, formats, offsets):
        """"""
        if isinstance(names, list):
            if len(names) == len(formats) == len(offsets):
                self.names.extend(names)
                self.formats.extend(formats)
                self.offsets.extend(offsets)
            else:
                raise RuntimeError("Lengths are not equal")
        else:
            self.names.append(names)
            self.formats.append(formats)
            self.offsets.append(offsets)

    @property
    def dict(self):
        """Return the dict representation"""
        return dict(names=self.names, formats=self.formats, offsets=self.offsets)

    @property
    def dtype(self):
        """return the dtype"""
        return np.dtype(self.dict)


def _read_header(fobj):
    # make emtpy header to fill
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
    # read initial header
    base_header = np.fromfile(fobj, dtype=header_dtype.dtype, count=1)
    fibsem_header = FIBSEMHeader(**dict(zip(base_header.dtype.names, base_header[0])))
    # now fobj is at position 34, return to 0
    fobj.seek(0, os.SEEK_SET)
    if fibsem_header.FileMagicNum != MAGIC_NUMBER:
        raise RuntimeError(
            f"FileMagicNum should be {MAGIC_NUMBER} but is {fibsem_header.FileMagicNum}"
        )

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
        # Read in AI channel scaling factors, (col#: AI#), (row#: offset, gain, 2nd order, 3rd order)
        header_dtype.update("Scaling", (">f4", (2, 4)), _scaling_offset)

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
            "Notes",  # Read in notes
        ],
        [">u1", ">u1", ">u1", ">u1", ">S200"],
        [151, 152, 153, 154, 180],
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
                # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
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
                # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
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

    if fibsem_header.FileVersion in {5, 6, 7, 8}:
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
                "MillingPIDMeasured",  # FIB milling PID measured (0:specimen, 1:beamdump)
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

    if fibsem_header.FileVersion in {6, 7}:
        header_dtype.update(
            [
                "Temperature",  # Temperature (F)
                "FaradayCupI",  # Faraday cup current (nA)
                "FIBSpecimenI",  # FIB specimen current (nA)
                "BeamDump1I",  # Beam dump 1 current (nA)
                "SEMSpecimenI",  # SEM specimen current (nA)
                "MillingYVoltage",  # Milling Y voltage (V)
                "FocusIndex",  # Focus index
                "FIBSliceNum",  # FIB slice #
            ],
            [">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">u4"],
            [850, 854, 858, 862, 866, 870, 874, 878],
        )
    if fibsem_header.FileVersion == 8:
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
    header = np.fromfile(fobj, dtype=header_dtype.dtype, count=1)
    fibsem_header = FIBSEMHeader(**dict(zip(header.dtype.names, header[0])))
    fibsem_header.to_native_types()

    return fibsem_header


def _read(path: str) -> FIBSEMData:
    """

    Read a single .dat file.

    Parameters
    ----------
    path : string denoting a path to a .dat file

    Returns : an instance of FIBSEMData
    -------

    """
    # Load raw_data data file 's' or 'ieee-be.l64' Big-ian ordering, 64-bit long data type
    with open(path, "rb") as fobj:
        fibsem_header = _read_header(fobj)
    # read data
    shape = (fibsem_header.YResolution,
             fibsem_header.XResolution,
             fibsem_header.ChanNum,)
    offset = 1024
    if fibsem_header.Eighbit == 1:
        dtype = ">u1"
    else:
        dtype = ">i2"

    try:
        raw_data = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=shape,
        )
    except ValueError:
        raw_data = np.zeros(dtype=dtype, shape=shape)

    raw_data = np.rollaxis(raw_data, 2)
    # Once read into the FIBSEMData structure it will be in memory, not memmap.
    return FIBSEMData(raw_data, fibsem_header)


# reading multiple files is handled upstream in fst.io.read
def read_fibsem(path: Union[str, Iterable[str]]):
    """

    Parameters
    ----------
    path : string or iterable of strings representing paths to .dat files.

    Returns a single FIBSEMDataset or an iterable of FIBSEMDatasets, depending on whether a single path or multiple
    paths are supplied as arguments.
    -------

    """
    if isinstance(path, str):
        return _read(path)
    elif isinstance(path, Iterable):
        return [_read(p) for p in path]
    else:
        raise ValueError("Path must be an instance of string or iterable of strings")


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

    else:
        if fibsem.header.FileVersion in {1, 2, 3, 4, 5, 6}:
            # scaled =

            if fibsem.header.AI1:
                detector_a = (
                    fibsem.header.Scaling[0, 0]
                    + fibsem[0] * fibsem.header.Scaling[0, 1]
                )
                if fibsem.header.AI2:
                    detector_b = (
                        fibsem.header.Scaling[1, 0]
                        + fibsem[1] * fibsem.header.Scaling[1, 1]
                    )
                    if fibsem.header.AI3:
                        detector_c = (
                            fibsem.header.Scaling[2, 0]
                            + fibsem[1] * fibsem.header.Scaling[1, 1]
                        )
        else:
            pass
    return detector_a, detector_b, scaled
