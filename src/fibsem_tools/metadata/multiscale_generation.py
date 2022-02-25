from pydantic import BaseModel
from enum import Enum
from typing import Optional, Tuple, Dict, Union, Any


class AccessMode(str, Enum):
    w = 'w'
    a = 'a'
    r = 'r'


MutableAccessMode = Union[AccessMode.w, AccessMode.a]
ImmutableAccessMode = AccessMode.r


class ChunkMode(str, Enum):
    auto = "auto"
    original = "original"


class MultiscaleMode(str, Enum):
    chained_memory = "chained_memory"
    chained_storage = "chained_storage"


class ContentType(str, Enum):
    scalar = "scalar"
    label = "label"


class Reducer(str, Enum):
    mode = "mode"
    mean = "mean"


class MultiscaleMetadataFlavor(str, Enum):
    neuroglancer = "neuroglancer"
    protoNGFF = "protoNGFF"


class DownsamplingDepth(str, Enum):
    all = "all"
    chunk_limited = "chunk_limited"


class WorkerSpec(BaseModel):
    num_workers: int
    num_cores: int
    memory: str


class ClusterType(str, Enum):
    dask_local = "dask_local"
    dask_lsf = "dask_lsf"


class ArrayStore(BaseModel):
    url: str
    chunks: Union[ChunkMode, Tuple[int, ...]]
    access_mode: AccessMode
    storage_options: Dict[str, Any] = {}


class DownsamplingSpec(BaseModel):
    method: Reducer
    factors: Tuple[int, ...]
    levels: Tuple[int, ...]
    chunks: Tuple[int, ...]


class ClusterSpec(BaseModel):
    deployment: ClusterType
    worker: WorkerSpec


class MultiscaleStorageSpec(BaseModel):
    source: ArrayStore
    destination: ArrayStore
    downsampling_spec: DownsamplingSpec
    cluster_spec: ClusterSpec
    logging_dir: Optional[str]


