from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import tensorstore as ts
import numpy as np
import json
from pathlib import Path
from xarray.core.dataarray import DataArray 

DRIVERS = {'n5', 'neuroglancer_precomputed'}
KVSTORE_DRIVERS = {'file', 'gcs'}

@dataclass
class StripNullFields():
    def asdict(self):
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                if hasattr(v, 'asdict'):
                    result[k] = v.asdict()
                elif isinstance(v, list):
                    result[k] = []
                    for element in v:
                        if hasattr(element, 'asdict'):
                            result[k].append(element.asdict)
                        else:
                            result[k].append(element)
                else:
                    result[k] = v
        return result

@dataclass
class KVStore(StripNullFields):
    driver: str
    path: str


@dataclass
class ImageScaleMetadata(StripNullFields):
    size: Sequence[int]
    resolution: Sequence[int]
    encoding: str
    chunk_size: Optional[Sequence[int]] = None
    chunk_sizes: Optional[Sequence[int]] = None
    key: Optional[str] = None
    voxel_offset: Optional[Sequence[int]] = None
    sharding: Optional[Dict[str, Any]] = None


@dataclass
class MultiscaleMetadata(StripNullFields):
    data_type: str
    num_channels: int
    type: str


@dataclass
class TensorStoreSpec(StripNullFields):
    driver: str
    kvstore: KVStore
    path: str
    scale_index: Optional[int] = 0
    scale_metadata: Optional[ImageScaleMetadata] = None
    multiscale_metadata: Optional[MultiscaleMetadata] = None


@dataclass
class PrecomputedMetadata:
    at_type: str
    volume_type: str
    data_type: str
    num_channels: str
    scales: Sequence[ImageScaleMetadata]


def parse_info(text: str) -> PrecomputedMetadata:
    json_data = json.loads(text)
    at_type = json_data['@type']
    volume_type = json_data['type']
    data_type = json_data['data_type']
    num_channels = json_data['num_channels']
    scales = [ImageScaleMetadata(**scale) for scale in json_data['scales']]

    return PrecomputedMetadata(at_type = at_type, 
                               volume_type=volume_type, 
                               data_type=data_type,
                               num_channels=num_channels,
                               scales=scales)

def PrecomputedFromDataArray(dataarray, 
                             path, 
                             encoding, 
                             volume_type, 
                             num_channels=1,
                             voxel_offset=None,
                             key=None, 
                             chunk_size=None,
                             **kwargs):
    assert len(dataarray.coords) == dataarray.ndim
    if voxel_offset is None:
       voxel_offset = (0,) * dataarray.ndim
        
    resolution = [abs(float(a[1].values - a[0].values)) for a in dataarray.coords.values()]
    tsa = TensorStoreArray(driver='neuroglancer_precomputed',
                           path=Path(path).parts[-1],
                           kvstore_driver='file',
                           kvstore_path=str(Path(path).parent),
                           encoding = encoding,
                           volume_type=volume_type,
                           num_channels=num_channels,
                           voxel_offset=voxel_offset,
                           resolution=resolution,
                           key=key,
                           template=dataarray,
                           chunk_size=chunk_size)
    print(tsa.spec['scale_metadata'])
    return tsa.open(**kwargs)

#todo: formally distinguish arrays from groups/containers
class TensorStoreArray:
    def __init__(
        self,
        driver,
        path,
        kvstore_driver,
        kvstore_path,
        encoding,
        num_channels,
        volume_type,
        voxel_offset,
        resolution,
        key=None,
        dtype=None,
        size=None,
        template=None,
        chunk_size=None,
        sharding=None,
        scale_index=None,
    ):
        '''
        A (relatively) friendly interface to tensorstore arrays. 
        '''

        if template is not None:
            if size is not None:
                raise ValueError("Must supply either size or template but not both.")
            if dtype is not None:
                raise ValueError("Must supply either dtype or template but not both.")
            
            size = template.shape
            dtype = template.dtype.name
            if chunk_size is None:
                if hasattr(template, "chunks"):
                    chunk_size = [c[0] for c in template.chunks]                

        if chunk_size is None:
            raise ValueError("Required argument `chunk_size` not supplied.")

        if dtype != 'uint8' and encoding == 'jpeg':
            raise ValueError(f'JPEG encoding only works for uint8 arrays. Your array is {dtype}')
        
        scale_metadata = ImageScaleMetadata(
            chunk_size=chunk_size,
            size=size,
            voxel_offset=voxel_offset,
            resolution=resolution,
            key=key,
            encoding=encoding,
            sharding=sharding,
        )

        kvstore = KVStore(driver=kvstore_driver, path=kvstore_path)

        multiscale_metadata = MultiscaleMetadata(
            data_type=dtype, num_channels=num_channels, type=volume_type
        )

        self.spec = TensorStoreSpec(driver, kvstore, path, scale_index=scale_index,scale_metadata=scale_metadata, multiscale_metadata=multiscale_metadata).asdict()
        
        
    def open(self, **kwargs):
        return ts.open(self.spec, **kwargs)


@dataclass
class NicerTensorStore:
    spec: Dict[str, Any]
    open_kwargs: Dict[str, Any]
    def __getitem__(self, slices):
        return ts.open(spec=self.spec, **self.open_kwargs).result()[slices]
    
    def __setitem__(self, slices, values):
        ts.open(spec=self.spec, **self.open_kwargs).result()[ts.d['channel'][0]][slices] = values
        return None

def prepare_tensorstore_from_pyramid(pyr: Sequence[DataArray], level_names: Sequence[str], jpeg_quality: int, output_chunks: Sequence[int], root_container_path: Path):
    store_arrays = []
    #sharding = {'@type': 'neuroglancer_uint64_sharded_v1',
    #       'preshift_bits': 9,
    #        'hash': 'identity',
    #        'minishard_index_encoding': 'gzip',
    #       'minishard_bits': 6,
    #       'shard_bits': 15}
    
    for p, ln in zip(pyr, level_names):
        res = [abs(float(p.coords[k][1] - p.coords[k][0])) for k in p.dims]
        spec: Dict[str, Any] = {'driver': 'neuroglancer_precomputed',
        'kvstore': {'driver': 'file', 
                    'path': str(Path(root_container_path).parent)},
        'path': root_container_path.parts[-1],
        'scale_metadata': {'size': p.shape,
        'resolution': res,
        'encoding': 'jpeg',
        'jpeg_quality': jpeg_quality,
        #'sharding': sharding,
        'chunk_size': output_chunks,
        'key': ln,
        'voxel_offset': (0, 0, 0)},
        'multiscale_metadata': {'data_type': p.dtype.name,
        'num_channels': 1,
        'type': 'image'}}
        try: 
            ts.open(spec=spec, open=True).result()
        except ValueError:
            try:
                ts.open(spec=spec, create=True).result()
            except ValueError:
                ts.open(spec=spec, create=True, delete_existing=True).result()

        nicer_array = NicerTensorStore(spec=spec, open_kwargs={'write': True})
        store_arrays.append(nicer_array)
    return store_arrays