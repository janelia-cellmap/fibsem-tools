from dataclasses import dataclass, asdict
from typing import Optional
import tensorstore as ts


@dataclass
class KVStore:
    driver: str
    path: str


@dataclass
class ImageScaleMetadata:
    size: list
    chunk_size: list
    voxel_offset: Optional[list]
    resolution: list
    encoding: str
    sharding: Optional[dict]


@dataclass
class MultiscaleMetadata:
    data_type: str
    num_channels: int
    type: str


@dataclass
class TensorStoreSpec:
    driver: str
    kvstore: KVStore
    path: str
    scale_metadata: ImageScaleMetadata
    multiscale_metadata: MultiscaleMetadata


import tensorstore as ts


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
        offset,
        resolution,
        *,
        dtype=None,
        size=None,
        template=None,
        chunk_size=None,
        sharding=None
    ):
    '''
    A (relatively) friendly interface to tensorstore arrays. 
    '''

        if template is not None:
            if size is not None and not np.array_equal(size, template.shape):
                raise ValueError("Must either specify size or data, not both")

            size = template.shape
            dtype = template.dtype.name
            if chunk_size is None:
                if hasattr(template, "chunksize"):
                    chunk_size = template.chunksize
                else:
                    raise ValueError("Required argument `chunk_size` not supplied")

        if chunk_size is None:
            raise ValueError("Required argument `chunk_size` not supplied.")

        scale_metadata = ImageScaleMetadata(
            chunk_size=chunk_size,
            size=size,
            voxel_offset=offset,
            resolution=resolution,
            encoding=encoding,
            sharding=sharding,
        )

        kvstore = KVStore(driver=kvstore_driver, path=kvstore_path)

        multiscale_metadata = MultiscaleMetadata(
            data_type=dtype, num_channels=num_channels, type=volume_type
        )

        self.spec = asdict(
            TensorStoreSpec(driver, kvstore, path, scale_metadata, multiscale_metadata)
        )

    def open(self, **kwargs):
        return ts.open(self.spec, **kwargs)
