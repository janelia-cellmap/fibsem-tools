from fibsem_tools.io import access, read_xarray, read
import numpy as np
import dask.array as da
from dask_janelia import get_cluster
import distributed
from distributed import Client, performance_report
import dask
import os
from fibsem_tools.io.multiscale import Multiscales, mode_reduce
from fibsem_tools.io.dask import ensure_minimum_chunksize
from xarray_multiscale import multiscale
from distributed import performance_report
from glob import glob
import toolz as tz
import click

def mean_reduce(v, **kwargs):
    return np.mean(v, **kwargs, dtype='float32')

def flip_y(d):
    reverser = slice(None, None, -1)
    d_new = d.isel({'y' : reverser})
    d_new.coords['y'] = d_new.coords['y'][reverser]
    return d_new

def remap_group_attrs(old_attrs):
    from copy import deepcopy
    new_attrs = deepcopy(old_attrs)
    for idx, value in enumerate(new_attrs['multiscales'][0]['datasets']):
        old_transform = value['transform']
        new_transform = old_transform.copy()
        for key in new_transform:
            new_transform[key] = old_transform[key][::-1]
        new_attrs['multiscales'][0]['datasets'][idx]['transform'] = new_transform
    return new_attrs

def remap_attrs(old_attrs):
    from copy import deepcopy
    new_attrs = deepcopy(old_attrs)
    old_transform = old_attrs['transform']
    new_transform = old_transform.copy()
    for key in new_transform:
        new_transform[key] = old_transform[key][::-1]
        new_attrs['transform'] = new_transform
    return new_attrs

def is_multiscale(group):
    return 'multiscales' in group.attrs

def get_upload_targets(group):
    results = {}
    for key, value in group.items():
        if hasattr(value, 'shape'):
            results[value.path] = value
        elif is_multiscale(value):
            results[value.path] = value['s0']
        else:
            results.update(get_upload_targets(value))
    return results


def make_multiscale(dataset: str, path: str, out_path: str, reference_path: str, dry: bool):
    read_chunks = (512,) * 3 
    store_chunk_map={'dense' : (64,) *3, 'sparse': (256,) * 3}
    scale_factors = (2,) * 3

    if reference_path:
        reference_coords = read_xarray(reference_path).coords
    else:
        reference_coords = None

    def to_upload(v):
        return True

    to_skip = lambda v: False
    to_flip = ()

    locking = False

    num_workers = 20
    # source_arrays = get_upload_targets(read(path))
    source_arrays  = {'labels/gt': path}

    for source_name, source in source_arrays.items():
        if not to_upload(source_name) or to_skip(source_name):
            click.echo(f'Skipping {source_name}')
        else:
            click.echo(f'Uploading {source_name} to {os.path.join(out_path, source_name)}')
            dest_name = source_name
            data = read_xarray(path, chunks=read_chunks, name=dest_name, storage_options={'normalize_keys': False})
            if source_name in to_flip:
                click.echo('flipping')
                data = flip_y(data)
            
            if reference_coords:
                data = data.assign_coords(reference_coords)
            
            if source_name.endswith('pred') or ('fibsem' in source_name) or ('raw' in source_name) or ('em' in source_name) or ('lm' in source_name):
                reducer = mean_reduce
                store_chunks = store_chunk_map['dense']
            else:
                reducer = mode_reduce
                store_chunks = store_chunk_map['sparse']
            
            scales = {f's{idx}' : v for idx,v in enumerate(multiscale(data, reducer, scale_factors, chained=True)[:5])}
            ms = Multiscales(name=dest_name, arrays=scales)

            if not locking:
                for k,v in ms.arrays.items():
                    v.data = ensure_minimum_chunksize(v.data, store_chunks)

            if not dry:
                lsf_kwargs={'walltime' : "24:00", "memory" : '30GB'}
                click.echo(f'reducing with {reducer.__name__}, saving {tuple(scales.keys())} to storage with {store_chunks=}')
                with get_cluster(threads_per_worker=2, lsf_kwargs=lsf_kwargs) as clust, Client(clust) as cl:
                    storage_group, storage_arrays, storage_op = ms.store(store=out_path, mode='a', storage_options={'normalize_keys' : False, 'profile' : 'COSEMPDSAdmin'}, chunks=store_chunks, locking=locking, client=cl)
                    click.echo(cl.cluster.dashboard_link)
                    cl.cluster.scale(num_workers)
                    results = cl.compute(dask.delayed(storage_op), sync=True)


@click.command()
@click.argument('dataset', type='str')
@click.argument('path', type='str')
@click.argument()
@click.argument()
def main():
    make_multiscale(dataset: str, path: str, out_path: str, reference_path: str, dry:bool)