from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple, Literal, Sequence, Any
import fsspec
import numpy as np
from xarray import DataArray
from dataclasses import asdict, dataclass
import json
from ..io.mrc import mrc_to_dask
from ..io import read
import dask.array as da
import dacite
from xarray_multiscale.metadata.util import SpatialTransform

CONTAINER_TYPES ={'mrc', 'n5', 'precomputed'}
DTYPE_FORMATS = {"uint16": "n5", "uint8": "precomputed", "uint64": "n5"}
CONTENT_TYPES = {"em", "lm", "prediction", "segmentation", "analysis"}
ContainerTypes = Literal['n5', 'precomputed', 'mrc']


@dataclass
class VolumeStorageSpec:
    kvStore: str
    containerType: ContainerTypes
    containerPath: str
    dataPath: str

    def toURI(self):
        return f'{self.kvStore}://{Path(self.containerPath).with_suffix("." + self.containerType).joinpath(self.dataPath)}'

    def __post_init__(self):
        if self.containerType not in CONTAINER_TYPES:
            raise ValueError(
                f"containerType must be one of {CONTAINER_TYPES}"
            )

@dataclass
class ContrastLimits:
    min: float
    max: float

    def __post_init__(self):
        if not self.min <= self.max:
            raise ValueError('min must be less than or equal to max.')


@dataclass
class DisplaySettings:
    contrastLimits: ContrastLimits
    color: str = 'white'
    invertColormap: bool = False

    @classmethod
    def fromDict(cls, d: Dict[str, Any]):
        return dacite.from_dict(cls, d)


@dataclass 
class DatasetView:
    datasetName: str
    name: str
    description: str
    position: Optional[Sequence[float]]
    scale: Optional[float]
    volumeKeys: Sequence[str]
 
    @classmethod
    def fromDict(cls, d: Dict[str, Any]):
        return dacite.from_dict(cls, d)


@dataclass
class MultiscaleSpec:
    reduction: str
    depth: int
    factors: Union[int, Sequence[int]]


@dataclass
class MeshSource:
    path: str
    name: str
    datasetName: str
    format: str


@dataclass
class VolumeSource:
    path: str
    name: str
    datasetName: str
    dataType: str
    dimensions: Sequence[float]
    transform: SpatialTransform
    contentType: str
    containerType: Optional[ContainerTypes]
    displaySettings: DisplaySettings
    description: str = ''
    version: str="0"
    tags: Optional[Sequence[str]] = None

    def __post_init__(self):
        assert self.contentType in CONTENT_TYPES
        assert len(self.version) > 0

    def toDataArray(self):
        if Path(self.path).suffix == ".mrc":
            array = mrc_to_dask(self.path, chunks=(1, -1, -1))
        else:
            r = read(self.path)
            array = da.from_array(r, chunks=r.chunks)
        coords = [
            DataArray(
                self.transform.translate[idx] + np.arange(array.shape[idx]) * self.transform.scale[idx],
                dims=ax,
                attrs= {'units': self.transform.units[idx]}
                )
            for idx, ax in enumerate(self.transform.axes)
        ]
        return DataArray(array, coords=coords, name=self.name)


    @classmethod
    def fromDict(cls, d: Dict[str, Any]):
        return dacite.from_dict(cls, d)



@dataclass
class DatasetIndex:
    name: str
    volumes: Sequence[VolumeSource]
    meshes: Sequence[MeshSource]
    views: Sequence[DatasetView]
    
    @classmethod
    def from_json(cls, fname: Union[str, Path], open_kwargs: dict = {}):
        with fsspec.open(str(fname), mode='rt', **open_kwargs) as fh:
            jblob = json.loads(fh.read())
        return cls(**jblob)
    
    def to_json(self, fname: Union[str, Path], open_kwargs: dict = {}) -> int:
        jblob = json.dumps(asdict(self))
        with fsspec.open(str(fname), mode='wt', **open_kwargs) as fh:
            result = fh.write(jblob)
        return result


@dataclass
class VolumeIngest:
    source: VolumeSource
    multiscaleSpec: MultiscaleSpec
    storageSpec: VolumeStorageSpec
    mutation: Optional[str] = None


@dataclass
class COSEMArrayAttrs:
    name: str
    transform: SpatialTransform

    @classmethod
    def fromDataArray(cls, data: DataArray) -> "COSEMArrayAttrs":
        name = data.name
        if name is not None:
            return cls(str(name), SpatialTransform.fromDataArray((data)))
        else: 
            raise ValueError('DataArray argument must have a valid name')


@dataclass
class OMEScaleAttrs:
    path: str 
    transform: SpatialTransform


@dataclass
class OMEMultiscaleAttrs:
    datasets: Sequence[OMEScaleAttrs]


@dataclass
class COSEMGroupAttrs:
    name: str
    multiscales: Sequence[OMEMultiscaleAttrs]


@dataclass
class N5PixelResolution:
    dimensions: Sequence[float]
    unit: str


@dataclass
class NeuroglancerGroupAttrs:
    # see https://github.com/google/neuroglancer/issues/176#issuecomment-553027775
    axes: Sequence[str]
    units: Sequence[str]
    scales: Sequence[Sequence[int]]
    pixelResolution: N5PixelResolution


@dataclass
class MultiscaleGroupAttrs:
    name: str
    multiscales: Sequence[OMEMultiscaleAttrs]
    axes: Sequence[str]
    units: Sequence[str]
    scales: Sequence[Sequence[int]]
    pixelResolution: N5PixelResolution


def makeN5ArrayAttrs(dimensions: Sequence[float], unit: str) -> Dict[str, N5PixelResolution]:
    return {'pixelResolution': N5PixelResolution(dimensions, unit)}


def makeMultiscaleGroupAttrs(name: str,
                            arrays: Sequence[DataArray], 
                            array_paths: Sequence[str], 
                            axis_order: str="F") -> MultiscaleGroupAttrs:
    
    assert len(arrays) == len(array_paths)
    cosemArrayAttrs = tuple(COSEMArrayAttrs.fromDataArray(a) for a in arrays)
    
    axis_indexer = slice(None)
    # neuroglancer wants the axes reported in fortran order
    if axis_order == "F":
        axis_indexer = slice(-1, None, -1)
        
    axes: Tuple[str] = arrays[0].dims[axis_indexer]
    scales = tuple(tuple(s.scale_factors)[axis_indexer] for s in arrays)
    coords_reordered = tuple(arrays[0].coords[k] for k in axes)
    units = tuple(d.units for d in coords_reordered)

    # we need this for neuroglancer
    pixelResolution = N5PixelResolution(dimensions=cosemArrayAttrs[0].transform.scale[axis_indexer], unit=units[0])
    multiscales = OMEMultiscaleAttrs(datasets=[OMEScaleAttrs(path=ap, transform=attr.transform) for ap, attr in zip(array_paths, cosemArrayAttrs)])

    result = MultiscaleGroupAttrs(name=name,
                                  multiscales=[multiscales], 
                                  axes=axes,
                                  units=units,
                                  scales=scales,
                                  pixelResolution=pixelResolution)
    return result


@dataclass
class CompositeArrayAttrs:
    name: str
    transform: SpatialTransform
    pixelResolution: N5PixelResolution

    @classmethod
    def fromDataArray(cls, data: DataArray):
        cosemAttrs = COSEMArrayAttrs.fromDataArray(data)
        pixelResolution = N5PixelResolution(cosemAttrs.transform.scale[::-1], unit=cosemAttrs.transform.units[0])
        return cls(cosemAttrs.name, cosemAttrs.transform, pixelResolution)