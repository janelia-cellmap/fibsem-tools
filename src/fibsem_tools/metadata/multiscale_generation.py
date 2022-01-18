from distributed.deploy import cluster
from numpy import array
from pydantic import BaseModel
from enum import Enum
import click
from typing import Optional, Tuple, Dict, Union, Literal, Any

from pydantic.errors import ArbitraryTypeError
from xarray.core.dataarray import DataArray
from fibsem_tools.io import read_xarray
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mean, windowed_mode
from fibsem_tools.io.io import AccessMode

from fibsem_tools.io.multiscale import Multiscales
from fibsem_tools.io.util import split_by_suffix
import fsspec
import json
import dask
from distributed import Client, performance_report
from datetime import datetime
import time
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WriteableAccessMode(str, Enum):
    w = "w"
    a = "a"


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
    storage_options: Dict[str, Any]
    chunks: Union[ChunkMode, Tuple[int, ...]]


class ReadableArrayStore(ArrayStore):
    access_mode: Literal["r"] = "r"


class WriteableArrayStore(ArrayStore):
    access_mode: Union[
        WriteableAccessMode, Tuple[WriteableAccessMode, WriteableAccessMode]
    ]


class DownsamplingSpec(BaseModel):
    method: Reducer
    factors: Tuple[int, ...]
    levels: Tuple[int, ...]
    chunks: Tuple[int, ...]


class ClusterSpec(BaseModel):
    deployment: ClusterType
    worker: WorkerSpec


class MultiscaleStorageSpec(BaseModel):
    source: ReadableArrayStore
    destination: WriteableArrayStore
    downsampling_spec: DownsamplingSpec
    cluster_spec: ClusterSpec
    logging_dir: Optional[str]


def prepare_multiscale_storage(
    source: str,
    source_chunks: Union[Tuple[int, ...], ChunkMode],
    dest: str,
    dest_chunks: Union[Tuple[int, ...], ChunkMode],
    dest_access_mode: Union[
        WriteableAccessMode, Tuple[WriteableAccessMode, WriteableAccessMode]
    ],
    downsampling_method: str,
    downsampling_factors: Tuple[int, ...],
    downsampling_levels: Tuple[int, ...],
    downsampling_chunks: Tuple[int, ...],
):

    chunk_mode = "minimum"
    source_xr = read_xarray(source, chunks=source_chunks, name=source)
    logger = logging.getLogger(__name__)
    logger.info(f"Found array {source_xr} at {source}")
    if downsampling_method == "mean":
        reducer = windowed_mean
    elif downsampling_method == "mode":
        reducer = windowed_mode
    else:
        raise ValueError(
            f'Invalid downsampling method. Must be one of ("mean", "mode"), got {downsampling_method}'
        )

    if isinstance(dest_access_mode, WriteableAccessMode):
        access_modes = (dest_access_mode,) * 2
    else:
        access_modes = dest_access_mode

    arrays = multiscale(
        source_xr,
        reducer,
        scale_factors=downsampling_factors,
        chunks=downsampling_chunks,
        chunk_mode=chunk_mode,
    )

    if len(downsampling_levels) == 0:
        downsampling_levels = tuple(range(len(arrays)))
    arrays = [arrays[idx] for idx in downsampling_levels]
    array_dict = {f"s{idx}": array for idx, array in zip(downsampling_levels, arrays)}
    logger.info(f"Prepared {len(array_dict)} arrays: {array_dict}")
    ms = Multiscales(name="foo", arrays=array_dict)
    store_group, store_arrays, storage = ms.store(
        dest, chunks=dest_chunks, access_modes=access_modes
    )

    data_volume = dask.utils.memory_repr(sum(a.nbytes for a in store_arrays))
    logger.info("Preparing to write to arrays:")
    for array in store_arrays:
        logger.info(array.info)
    logger.info(f"Total data volume: {data_volume}")
    return storage


@click.command()
@click.argument("config_json", type=str, required=True)
@click.option("--dry", type=bool)
@click.option("--scheduler", type=str)
def main(config_json: str, dry: bool = False, scheduler: str = ""):

    now_str = datetime.now().strftime("%Y:%m:%d:%H:%M:%S")

    with fsspec.open(config_json) as fh:
        json_blob = json.load(fh)

    spec = MultiscaleStorageSpec(**json_blob)

    if not os.path.exists(spec.logging_dir):
        os.makedirs(spec.logging_dir)

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(
                spec.logging_dir, f"multiscale_generation_{now_str}.log"
            )
        )
    )
    logger.addHandler(logging.StreamHandler())
    logger.info(f"Loaded MultiscaleStorageSpec: {spec.json(indent=2)}")

    store_op = prepare_multiscale_storage(
        source=spec.source.url,
        source_chunks=spec.source.chunks,
        dest=spec.destination.url,
        dest_chunks=spec.destination.chunks,
        dest_access_mode=spec.destination.access_mode,
        downsampling_method=spec.downsampling_spec.method,
        downsampling_factors=spec.downsampling_spec.factors,
        downsampling_levels=spec.downsampling_spec.levels,
        downsampling_chunks=spec.downsampling_spec.chunks,
    )

    if spec.cluster_spec.deployment == "dask_local":
        from distributed import LocalCluster

        clusterClass = LocalCluster
    elif spec.cluster_spec.deployment == "dask_lsf":
        from dask_jobqueue import LSFCluster
        from functools import partial

        clusterClass = partial(
            LSFCluster,
            ncpus=spec.cluster_spec.worker.num_cores,
            mem=f"{15 * spec.cluster_spec.worker.num_cores}GB",
            processes=spec.cluster_spec.worker.num_cores,
        )
        clusterClass.__name__ = LSFCluster.__name__

    if not dry:
        logger.info(
            f"Creating an instance of {clusterClass.__name__} and scaling to {spec.cluster_spec.worker.num_workers} workers"
        )
        with clusterClass() as clust, Client(clust) as cl, performance_report(
            os.path.join(
                spec.logging_dir, f"dask_distributed_perf_report_{now_str}.html"
            )
        ):
            cl.cluster.scale(spec.cluster_spec.worker.num_workers)
            logger.info(f"Cluster dashboard url: {cl.cluster.dashboard_link}")
            logger.info(f"Begin saving multiscale data to {spec.destination.url}")
            start = time.time()
            futures = cl.compute(store_op)
            results = cl.gather(futures)
            end = time.time()
            logger.info(f"Done saving multiscale data after {end - start}s")


if __name__ == "__main__":
    main()
