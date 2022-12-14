from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union, List
import tensorstore as ts
import json
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path
from xarray.core.dataarray import DataArray
import fsspec
import os
from dask.array.core import normalize_chunks
from dask.array import map_blocks

from fibsem_tools.io.util import split_by_suffix



class KVStoreDriver(str, Enum):
    http = "http"
    file = "file"
    gcs = "gcs"
    memory = "memory"
    neuroglancer_uint64_sharded = "neuroglancer_uint64_sharded"


class ChunkedStorageDriver(str, Enum):
    n5 = "n5"
    zarr = "zarr"
    neuroglancer_precomputed = "neuroglancer_precomputed"


class KVStore(BaseModel):
    driver: KVStoreDriver
    path: str


class Sharding(BaseModel):
    preshift_bits: int
    hash: str
    minishard_bits: int
    shard_bits: int
    minishard_index_encoding: str = "raw"
    data_index_encoding: str = "raw"
    type: str = Field("neuroglancer_uint64_sharded_v1", alias="@type")


class ScaleMetadata(BaseModel):
    size: Optional[Sequence[int]]
    encoding: Optional[str]
    chunk_size: Optional[List[int]]
    resolution: Optional[Sequence[float]]
    key: Optional[str] = None
    voxel_offset: Optional[Sequence[int]] = None
    jpeg_quality: Optional[int] = None
    sharding: Optional[Sharding] = None

    def __post_init__(self):
        if self.resolution:
            self.resolution = [float(r) for r in self.resolution]
            if self.key == None:
                self.key = "_".join(map(str, self.resolution))


class MultiscaleMetadata(BaseModel):
    type: str
    data_type: str
    num_channels: int


class TensorStoreSpec(BaseModel):
    driver: ChunkedStorageDriver
    kvstore: KVStore
    path: str
    scale_index: Optional[int] = 0
    scale_metadata: Optional[ScaleMetadata] = None
    multiscale_metadata: Optional[MultiscaleMetadata] = None


class PrecomputedMetadata(BaseModel):
    type: str
    data_type: str
    num_channels: int
    scales: Sequence[ScaleMetadata]
    type: Optional[str] = Field(None, alias='@type')

def PrecomputedFromDataArray(
    array,
    path,
    encoding,
    volume_type,
    num_channels=1,
    voxel_offset=None,
    key=None,
    chunk_size=None,
    **kwargs,
):
    assert len(array.coords) == array.ndim
    if voxel_offset is None:
        voxel_offset = (0,) * aarray.ndim

    resolution = [
        abs(float(a[1].values - a[0].values)) for a in array.coords.values()
    ]
    tsa = TensorStoreArray(
        driver="neuroglancer_precomputed",
        path=Path(path).parts[-1],
        kvstore_driver="file",
        kvstore_path=str(Path(path).parent),
        encoding=encoding,
        volume_type=volume_type,
        num_channels=num_channels,
        voxel_offset=voxel_offset,
        resolution=resolution,
        key=key,
        template=array,
        chunk_size=chunk_size,
    )
    return tsa.open(**kwargs).result()


# todo: formally distinguish arrays from groups/containers
class TensorStoreArray:
    def __init__(
        self,
        driver,
        path,
        kvstore_driver,
        kvstore_path,
        key,
        encoding=None,
        num_channels=None,
        volume_type=None,
        resolution=None,
        jpeg_quality=None,
        voxel_offset=None,
        dtype=None,
        size=None,
        template=None,
        chunk_size=None,
        sharding=None,
        scale_index=None,
    ):
        """
        A (relatively) friendly interface to tensorstore arrays.
        """

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
                else:
                    raise ValueError("Required argument `chunk_size` not supplied.")

            if dtype != "uint8" and encoding == "jpeg":
                raise ValueError(
                    f"JPEG encoding only works for uint8 arrays. You specified a dtype of {dtype}"
                )

        scale_metadata = ScaleMetadata(
            chunk_size=chunk_size,
            size=size,
            voxel_offset=voxel_offset,
            resolution=resolution,
            key=key,
            jpeg_quality=jpeg_quality,
            encoding=encoding,
            sharding=sharding,
        )

        kvstore = KVStore(driver=kvstore_driver, path=kvstore_path)
        
        if (dtype is not None and num_channels is not None and type is not None): 
            multiscale_metadata = MultiscaleMetadata(
                data_type=dtype, num_channels=num_channels, type=volume_type
            )

        self.spec = TensorStoreSpec(
            driver,
            kvstore,
            path,
            scale_index=scale_index,
            scale_metadata=scale_metadata,
            multiscale_metadata=multiscale_metadata,
        ).dict()

    def __repr__(self) -> str:
        return str(self.spec)

    def open(self, **kwargs):
        return ts.open(self.spec, **kwargs)


def access_precomputed(
    store_path: str,
    key: str,
    mode: str,
    array_type=None,
    dtype=None,
    num_channels=None,
    shape=None,
    resolution=None,
    encoding=None,
    chunks=None,
    jpeg_quality=None,
    voxel_offset=None,
    scale_index=None,
) -> TensorStoreArray:
    driver = "neuroglancer_precomputed"

    kvstore_driver, _store_path = fsspec.core.split_protocol(store_path)
    if kvstore_driver == None:
        kvstore_driver = "file"
        kvstore_path = "/"
        # remove the leading slash after making the absolute path
        _store_path = os.path.abspath(_store_path)[1:]
    else:
        kvstore_path = _store_path.split(os.path.sep)[0]

    if not any(x == kvstore_driver for x in  KVStoreDriver):
        raise ValueError(
            f"File system protocol {kvstore_driver} is not supported by tensorstore."
        )

    info_path = os.path.join(store_path, "info")

    if mode == "r":
        with fsspec.open(info_path) as fh:
            json_data = json.loads(fh.read())
            precomputed_metadata = parse_info(json_data)
            scale_matches = [scale.key == key for scale in precomputed_metadata.scales]
            if not any(scale_matches):
                raise ValueError(
                    "Could not find key: {key} in info file at {info_path}"
                )
            else:
                scale_index = scale_matches.index(True)
                scale_meta = precomputed_metadata.scales[scale_index]
    else:
        scale_meta = ScaleMetadata(
            size=shape,
            resolution=resolution,
            encoding=encoding,
            chunk_size=chunks,
            key=key,
            voxel_offset=voxel_offset,
            jpeg_quality=jpeg_quality,
        )
        precomputed_metadata = PrecomputedMetadata(
            type=array_type,
            data_type=dtype,
            num_channels=num_channels,
            scales=[scale_meta],
        )

    if mode == "r":
        read = True
        # So cool that tensorstore errors when these are set to False for reading...
        write = None
        create = None
        delete_existing = None
    elif mode == "a":
        read = True
        write = True
        create = True
        delete_existing = False
    elif mode == "rw":
        read = True
        write = True
        create = True
        delete_existing = True
    elif mode == "w":
        read = False
        write = True
        create = True
        delete_existing = True
    elif mode == "w-":
        read = False
        write = True
        create = True
        delete_existing = False
    else:
        raise ValueError('Mode must be "r", "rw", "a", "w", or "w-"')

    tsa = TensorStoreArray(
        driver=driver,
        path=_store_path,
        kvstore_path=kvstore_path,
        kvstore_driver=kvstore_driver,
        encoding=scale_meta.encoding,
        scale_index=scale_index,
        key=scale_meta.key,
        num_channels=precomputed_metadata.num_channels,
        volume_type=precomputed_metadata.type,
        dtype=precomputed_metadata.data_type,
        resolution=scale_meta.resolution,
        size=scale_meta.size,
        chunk_size=scale_meta.chunk_size,
        jpeg_quality=jpeg_quality,
    )
    return tsa.open(
        read=read, write=write, create=create, delete_existing=delete_existing
    ).result()


def precomputed_to_dask(
    urlpath: str, chunks: Union[str, Sequence[int]], channel: int = 0
):
    store_path, key, _ = split_by_suffix(urlpath, (".precomputed",))
    tsa = access_precomputed(store_path, key, mode="r")[ts.d["channel"][channel]]
    shape = tuple(tsa.shape)
    dtype = tsa.dtype.numpy_dtype
    if chunks == "original":
        chunks = tsa.spec().to_json()["scale_metadata"]["chunk_size"]
    _chunks = normalize_chunks(chunks, shape, dtype=dtype)

    def chunk_loader(store_path, key, block_info=None):
        idx = tuple(slice(*idcs) for idcs in block_info[None]["array-location"])
        tsa = access_precomputed(store_path, key, mode="r")[ts.d["channel"][channel]]
        result = tsa[idx].read().result()
        return result

    arr = map_blocks(chunk_loader, store_path, key, chunks=_chunks, dtype=dtype)
    return arr