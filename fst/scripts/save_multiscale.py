
from dataclasses import asdict, dataclass
from fst.io.tensorstore import prepare_tensorstore_from_pyramid
from fst.attrs.attrs import CompositeArrayAttrs, makeMultiscaleGroupAttrs
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Optional, List
import numpy as np
from xarray import DataArray
from distributed import Client
from fst.attrs import DatasetIndex, VolumeSource, VolumeIngest, MultiscaleSpec, VolumeStorageSpec, CompositeArrayAttrs
from fst.pyramid import lazy_pyramid, mode_reduce, blocked_pyramid, blocked_store
from dask_janelia import auto_cluster
from fst.io import access_multiscale
from tqdm import tqdm

def save_dataset_metadata(source: VolumeSource, index: DatasetIndex):
    index.volumes[source.path] = source
    index.to_json(index_file)    
    return 0

def flip_axis(arr: DataArray, axis: str) -> DataArray:
    arr2 = arr.copy()
    idx = arr2.dims.index(axis)
    arr2.data = np.flip(arr2.data, idx)
    return arr2

reducers: Dict[str, Callable[[Any], Any]] = {'mean' : np.mean, 'mode': mode_reduce}
mutations: Dict[str, Callable[[Any], Any]] = {'flip_y': lambda v: flip_axis(v, 'y')}
ndim = 3
metadata_only=False 
save_chunks=(96 * 2,) * ndim
output_chunks=(96,) * ndim,
scale_factors = (2,) * ndim,
blocked_pyramid_size=(96 * 8, -1, -1)
num_workers = 100
output_dir = '/nrs/cosem/davis/s3'

# mongodb variables
un = 'root'
pw = 'root'
addr = 'cosem.int.janelia.org'
db = 'VolumeSources'
catalog = 'to_ingest'


# update views
# index.views = views[source.datasetName]

@dataclass
class MultiscaleSavePlan:
    mode: str 
    save_chunks: Tuple[int, ...]
    output_chunks: Tuple[int, ...]
    blocked_pyramid_size: Tuple[int, ...]
    jpegQuality: int = 90


def get_sources_from_mongodb(query: Dict[str,str]) -> List[VolumeSource]:
    from pymongo import MongoClient

    # insert each element in the list into the `datasets` collection on our MongoDB instance
    with MongoClient(f'mongodb://{un}:{pw}@{addr}') as client:
        sources = tuple(client[db][catalog].find(query))

    sources_cleaned: List[VolumeSource] = []
    for s in sources:
        s.pop('_id')
        sources_cleaned.append(VolumeSource.fromDict(s)) 
    
    return sources_cleaned


def makeVolumeIngest(source: VolumeSource, output_path: str) -> VolumeIngest:
    mutation: Optional[str] = None
    reduction = 'mean'
    dataPath = ''
    containerPath = ''
    
    if source.dataType == 'uint8' and source.contentType == 'em':
        containerType = 'precomputed'
        dataPath = ''
        containerPath = str(Path(output_path) / source.datasetName / 'neuroglancer' / f'{source.name}.precomputed')
    else:
        containerType = 'n5'
        containerPath = str(Path(output_path) / source.datasetName / f'{source.datasetName}.n5')
        if source.contentType == 'em':
            dataPath = str(Path('em') / source.name)
        elif source.contentType in {'prediction', 'gt', 'segmentation'}:
            dataPath = str(Path('labels') / source.name)
        else:
            raise ValueError(f'content type {source.contentType} cannot be dispatched to a path')
    storageSpec = VolumeStorageSpec('file', containerType, containerPath, dataPath)
    
    if source.contentType in ('segmentation', 'prediction'):
        mutation = 'flip_y' 

    if source.contentType == 'segmentation':
        reduction = 'mode' 
    multiscaleSpec = MultiscaleSpec(reduction, 5, (2,2,2))

    ingest = VolumeIngest(source, multiscaleSpec, storageSpec, mutation)
    return ingest


def build_index(path: str):
    dataset_name = Path(path).name
    n5_volumes = filter(Path.is_dir, (Path(path) / f'{dataset_name}.n5').glob('*/*'))
    precomputed_volumes = filter(Path.is_dir, (Path(path) / 'neuroglancer').glob('*/*.precomputed'))
    volumes = (*n5_volumes, *precomputed_volumes)
    
    if len(volumes) < 1:
        raise FileNotFoundError(f'No volumes found in {path}')

    query = {'datasetName' : dataset_name}
    sources = get_sources_from_mongodb(query)
    if len(sources) < 1:
        raise ValueError(f'No sources found in the database using query {query}')
    source_names = [s.name for s in sources]
    volume_names = [v.stem for v in volumes]
    for idx, v in enumerate(volume_names):
        if v not in source_names:
            print(f'Warning: could not find an entry for {volumes[idx]} in the database')

    index_file = Path(path) / 'index.json'

    if not index_file.exists():
        DatasetIndex(name=dataset_name, volumes={}, views=[]).to_json(index_file)

    index = DatasetIndex.from_json(index_file)        
    # update views
    index.views = views


    return sources


def ingest_source(ingest: VolumeIngest, save_plan: MultiscaleSavePlan):
    source = ingest.source
    
    # this controls the order of linearized dimensional quantities like pyramid_block_size
    axis_order = slice(None)

    # the default mutation is the identity transformation
    mutation = mutations.get(ingest.mutation, lambda v: v)
    darr: DataArray = mutation(source.toDataArray())
    array_attrs = CompositeArrayAttrs.fromDataArray(darr)
    darr.attrs.update(asdict(array_attrs))
    darr.data = darr.data.rechunk(save_plan.save_chunks)
    if ingest.storageSpec.containerType == 'precomputed':
        darr = darr.T
        axis_order = slice(-1, None, -1)
    
    reducer = reducers[ingest.multiscaleSpec.reduction]
    multiscale_depth = ingest.multiscaleSpec.depth
    scale_factors = ingest.multiscaleSpec.factors

    # full_path = Path(output_dir) / Path(source.datasetName) / source.destPath

    pyr = lazy_pyramid(darr, 
                       reduction=reducer, 
                       scale_factors=scale_factors, 
                       max_depth=multiscale_depth)                    
    level_names = [f's{idx}' for idx in range(len(pyr))]
    container_path = Path(output_dir) / ingest.storageSpec.containerPath
    
    if ingest.storageSpec.containerType == 'n5':
        group_path = ingest.storageSpec.dataPath
        print(f'Preparing the store {Path(container_path) / group_path}')
        group_attrs = asdict(makeMultiscaleGroupAttrs(darr.name, pyr, level_names))
        store_group, store_arrays = access_multiscale(container_path,
                                                    group_path, 
                                                    pyr, 
                                                    array_paths=level_names,
                                                    array_chunks=save_plan.output_chunks,
                                                    attr_factory = lambda v: asdict(CompositeArrayAttrs.fromDataArray(v)))            
        store_group.attrs.put(group_attrs)
    elif ingest.storageSpec.containerType == 'precomputed':
        print(f'Preparing the store {container_path}')
        if darr.dtype.name not in {'uint8'}:
            raise ValueError('Only uint8 supported at this time')
        
        store_group = None
        store_arrays = prepare_tensorstore_from_pyramid(pyr, level_names, save_plan.jpegQuality, save_plan.output_chunks, container_path)

    if save_plan.mode == 'w':
        with auto_cluster(walltime='8:00') as clust, Client(clust) as cl:
            print(cl.cluster.dashboard_link)
            pyr_b = blocked_pyramid(darr, 
                                    block_size=save_plan.blocked_pyramid_size[axis_order], 
                                    reduction=reducer, 
                                    scale_factors=scale_factors[axis_order], 
                                    max_depth=multiscale_depth)

            stores = blocked_store(pyr_b, store_arrays, chunks=save_plan.output_chunks)                                            
            #cl.cluster.scale(num_workers)
            cl.cluster.adapt()
            for slab_idx, store in tqdm(enumerate(stores)):
                cl.compute(store).result()
                progress = f'{slab_idx + 1} / {len(stores)}'
                if ingest.storageSpec.containerType == 'n5':
                    store_group.attrs['creation_progress'] = progress
        

