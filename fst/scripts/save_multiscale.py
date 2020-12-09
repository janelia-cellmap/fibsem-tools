from dataclasses import asdict, dataclass
from fst.io.tensorstore import prepare_tensorstore_from_pyramid
from fst.attrs.attrs import CompositeArrayAttrs, DatasetView, MeshSource, makeMultiscaleGroupAttrs
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Optional, List, TypeVar
import numpy as np
import dask 
dask.config.set({'optimization.fuse.ave-width': 50})
from xarray import DataArray
from distributed import Client
from fst.attrs import (
    DatasetIndex,
    VolumeSource,
    VolumeIngest,
    MultiscaleSpec,
    VolumeStorageSpec,
    CompositeArrayAttrs,
)
from fst.pyramid import lazy_pyramid, mode_reduce, blocked_pyramid, blocked_store
from dask_janelia import auto_cluster
from fst.io import access_multiscale, read
from tqdm import tqdm
import click
from pymongo import MongoClient
from dataclasses import replace
from fsspec import filesystem

def infer_container_type(path: str) -> str:
    if Path(path).suffix == ".mrc":
        containerType = "mrc"
    elif any(map(lambda v: (Path(v).suffix == ".n5"), Path(path).parts)):
        containerType = "n5"
    elif any(map(lambda v: (Path(v).suffix == ".precomputed"), Path(path).parts)):
        containerType = "precomputed"
    else:
        raise ValueError(f"Could not infer container type from path {path}")

    return containerType


def save_dataset_metadata(source: VolumeSource, index: DatasetIndex):
    index.volumes[source.path] = source
    index.to_json(index_file)
    return 0


def flip_axis(arr: DataArray, axis: str) -> DataArray:
    arr2 = arr.copy()
    idx = arr2.dims.index(axis)
    arr2.data = np.flip(arr2.data, idx)
    return arr2


reducers: Dict[str, Callable[[Any], Any]] = {"mean": np.mean, "mode": mode_reduce}
mutations: Dict[str, Callable[[Any], Any]] = {"flip_y": lambda v: flip_axis(v, "y")}
ndim = 3
access_modes = ('write', 'metadata')
save_chunks = (96 * 2,) * ndim
output_chunks = (96,) * ndim
scale_factors = (2,) * ndim
blocked_pyramid_size = (96 * 4, -1, -1)
num_workers = 100
output_dir = "/nrs/cosem/davis/s3"

# mongodb variables
un = "root"
pw = "root"
addr = "cosem.int.janelia.org"
db = "sources"
mongo_addr = f"mongodb://{un}:{pw}@{addr}"


@dataclass
class MultiscaleSavePlan:
    overwrite: bool
    save_chunks: Tuple[int, ...]
    output_chunks: Tuple[int, ...]
    blocked_pyramid_size: Tuple[int, ...]
    jpegQuality: int = 90

T = TypeVar('T')

def typed_list_from_mongodb(mongo_addr: str, db: str, cls: T, query: Dict[str, str]) -> List[T]:
    with MongoClient(mongo_addr) as client:
        retrieved = client[db][cls.__name__].find(query)
    results = []
    for r in retrieved:
        r.pop('_id')
        results.append(cls(**r))
    return results


def makeVolumeIngest(source: VolumeSource, output_path: str) -> VolumeIngest:
    mutation: Optional[str] = None
    reduction = "mean"
    dataPath = ""
    containerPath = ""

    if source.dataType == "uint8" and source.contentType == "em":
        containerType = "precomputed"
        dataPath = ""
        containerPath = str(
            Path(output_path)
            / source.datasetName
            / "neuroglancer" 
            / "em" 
            / f"{source.name}.precomputed"
        )
    else:
        containerType = "n5"
        containerPath = str(
            Path(output_path) / source.datasetName / f"{source.datasetName}.n5"
        )
        if source.contentType == "em":
            dataPath = str(Path("em") / source.name)
        elif source.contentType in {"prediction", "gt", "segmentation", "analysis"}:
            dataPath = str(Path("labels") / source.name)
        else:
            raise ValueError(
                f"content type {source.contentType} cannot be dispatched to a path"
            )
    storageSpec = VolumeStorageSpec("file", containerType, containerPath, dataPath)

    if source.contentType != "em" and source.datasetName in ("jrc_hela-1", "jrc_hela-2", 'jrc_hela-3', 'jrc_jurkat-1', 'jrc_macrophage-2'):
        mutation = "flip_y"

    if source.contentType == "segmentation":
        reduction = "mode"
    multiscaleSpec = MultiscaleSpec(reduction, 5, (2, 2, 2))

    ingest = VolumeIngest(source, multiscaleSpec, storageSpec, mutation)
    return ingest


def build_index(URL: str):
    try:
        fs_protocol, root = URL.split('://')
    except ValueError:
        print(f'Your input {URL} could not be split into a protocol and a path')
        raise
    fs = filesystem(fs_protocol)
    dataset_name = Path(root).name

    n5_paths = tuple(filter(fs.isdir, fs.glob(root +  '/*n5*/*/*')))
    precomputed_paths = tuple(filter(fs.isdir, fs.glob(root +  '/neuroglancer/em/*')))
    mesh_paths = tuple(filter(fs.isdir, fs.glob(root +  '/neuroglancer/mesh/*')))

    output_mesh_sources = [MeshSource(str(Path(mp).relative_to(root)), Path(mp).stem, dataset_name, 'neuroglancer_legacy_mesh') for mp in mesh_paths]

    volume_paths = (*n5_paths, *precomputed_paths)
    volume_path_stems = tuple(Path(s).stem for s in volume_paths)
    
    query = {"datasetName": dataset_name}
    db_volume_sources: List[VolumeSource] = typed_list_from_mongodb(mongo_addr, db, VolumeSource, query)
    if len(db_volume_sources) < 1:
        raise ValueError(f"No sources found in the database using query {query}")
    
    db_source_dict = {s.name: s for s in db_volume_sources}
    output_volume_sources: List[VolumeSource] = []
    for sk, sv in db_source_dict.items():
        if sk not in volume_path_stems:
            print(f'Warning: could not find an extant volume on the filesystem matching this VolumeSource from the database: {sv}. This volume will not be added to the dataset index.')
        else:
            input_source: VolumeSource = replace(sv)
            vol_path = volume_paths[volume_path_stems.index(sk)]
            input_source.path = str(Path(vol_path).relative_to(root))
            input_source.containerType = infer_container_type(vol_path)
            output_volume_sources.append(input_source)

    db_views: List[DatasetView] = []
    db_views = typed_list_from_mongodb(mongo_addr, db, DatasetView, query)
    
    accepted_views = []
    for v in db_views: 
        missing = set(v.volumeKeys) - set(db_source_dict.keys())
        if len(missing) > 0:
            print(f'This view contains volumes: {missing} that could not be found in the volume source database and thus will not be included in the index: {v}')
        else:
            accepted_views.append(v)

    index = DatasetIndex(name=dataset_name, volumes=output_volume_sources, meshes=output_mesh_sources, views=accepted_views)

    return index


def ingest_source(ingest: VolumeIngest, save_plan: MultiscaleSavePlan, num_workers: int):
    source = ingest.source

    # this controls the order of linearized dimensional quantities like pyramid_block_size
    axis_order = slice(None)

    # the default mutation is the identity transformation
    print(f'Using mutation `{ingest.mutation}`')
    mutation = mutations.get(ingest.mutation, lambda v: v)
    darr: DataArray = mutation(source.toDataArray())
    array_attrs = CompositeArrayAttrs.fromDataArray(darr)
    darr.attrs.update(asdict(array_attrs))
    darr.data = darr.data.rechunk(save_plan.save_chunks)
    if ingest.storageSpec.containerType == "precomputed":
        darr = darr.T
        axis_order = slice(-1, None, -1)

    reducer = reducers[ingest.multiscaleSpec.reduction]
    multiscale_depth = ingest.multiscaleSpec.depth
    scale_factors = ingest.multiscaleSpec.factors

    # full_path = Path(output_dir) / Path(source.datasetName) / source.destPath

    pyr = lazy_pyramid(
        darr, reduction=reducer, scale_factors=scale_factors, max_depth=multiscale_depth
    )
    level_names = [f"s{idx}" for idx in range(len(pyr))]
    container_path = Path(output_dir) / ingest.storageSpec.containerPath

    if ingest.storageSpec.containerType == "n5":
        group_path = ingest.storageSpec.dataPath
        print(f"Preparing the store {Path(container_path) / group_path}")
        group_attrs = asdict(makeMultiscaleGroupAttrs(darr.name, pyr, level_names))
        store_group, store_arrays = access_multiscale(
            container_path,
            group_path,
            pyr,
            array_paths=level_names,
            array_chunks=save_plan.output_chunks,
            attr_factory=lambda v: asdict(CompositeArrayAttrs.fromDataArray(v)),
        )
        store_group.attrs.put(group_attrs)
    elif ingest.storageSpec.containerType == "precomputed":
        print(f"Preparing the store {container_path}")
        if darr.dtype.name not in {"uint8"}:
            raise ValueError("Only uint8 supported at this time")

        store_group = None
        store_arrays = prepare_tensorstore_from_pyramid(
            pyr,
            level_names,
            save_plan.jpegQuality,
            save_plan.output_chunks,
            container_path,
        )

    if save_plan.overwrite:
        with auto_cluster(walltime="8:00") as clust, Client(clust) as cl:
            print(cl.cluster.dashboard_link)
            pyr_b = blocked_pyramid(
                darr,
                block_size=save_plan.blocked_pyramid_size[axis_order],
                reduction=reducer,
                scale_factors=scale_factors[axis_order],
                max_depth=multiscale_depth,
            )

            stores = blocked_store(pyr_b, store_arrays, chunks=save_plan.output_chunks)
            cl.cluster.scale(num_workers)
            for store in tqdm(stores):
                cl.compute(store).result()


def check_initialized_chunks(path: str) -> bool:
    r = read(path)
    # i check whether nchunks is less than or equal to initialized chunks because it seems that the zarr python library
    # is a little generous with what it considers a chunk, and thus it's possible to have more chunks initialized than total chunks,
    # e.g. if some .nfsblablabla files are floating around from other processes
    return all(i.nchunks <= i.initialized for n,i in r.items())


@click.command()
@click.option("-q", "--query", required=True, type=str)
@click.option("-w", "--workers", required=False, type=int, default=0)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--metadata/--no-metadata", default=False)
def ingest_source_cli(query: str, workers: int, overwrite: bool, metadata: bool):
    if overwrite and metadata:
        print('Warning: you have selected overwrite AND metadata operating modes. The metadata operating mode takes precedence, and thus incomplete containers will not be completed.')
        overwrite = False
    query_result = get_sources_from_mongodb(eval(query))
    if len(query_result) == 0:
        raise ValueError(f'Could not find a single dataset using the query {query}')
    sources = query_result
    for source in sources:
        ingest = makeVolumeIngest(source, output_dir)
        plan = MultiscaleSavePlan(overwrite=overwrite, save_chunks=save_chunks, output_chunks=output_chunks, blocked_pyramid_size=blocked_pyramid_size)
        pth = (Path(ingest.storageSpec.containerPath) / ingest.storageSpec.dataPath)
        incomplete = False
        if not pth.exists():
            incomplete = True
            print(f'Nothing found at {pth}. Multiscale volume will be created.')
            if not metadata:
                plan.overwrite = True
        elif ingest.storageSpec.containerType == 'n5' and not metadata:
            incomplete = not check_initialized_chunks(str(pth))
            if incomplete:
                print(f'Incomplete multiscale volume found at {pth}. Multiscale volume will be created.')
                plan.overwrite = True
                
        print(f'Ingesting store at {pth}...')
        ingest_source(ingest, plan, num_workers=workers)
        
if __name__ == '__main__':
    ingest_source_cli()