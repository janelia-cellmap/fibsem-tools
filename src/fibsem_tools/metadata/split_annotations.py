import os
from typing import Optional
import click
from fibsem_tools import read_xarray
from fibsem_tools.io.zarr import access_zarr
import numpy as np
import fibsem_tools.metadata.groundtruth as gt
from xarray import DataArray

from pathlib import Path
import zarr
from fibsem_tools.io.util import split_by_suffix
from numcodecs import Blosc
from xarray_ome_ngff import get_adapters

ome_adapters = get_adapters('0.4')
out_chunks = (256,) * 3

annotation_type = gt.SemanticSegmentation(
        encoding={
            'absent': 0, 
            'present': 1
            }
        )


def split_annotations(source: str, dest: str, name: Optional[str]=None) -> zarr.Group:
    tree = read_xarray(source)
    # todo: don't assume the biggest array is named s0!
    array_name = 's0'
    source_xr = tree[array_name].data
    
    ome_meta = ome_adapters.multiscale_metadata([source_xr], 
                                                [array_name])
    if name is None:
        crop_name = Path(source).stem
    else:
        crop_name = name
    
    uniques = np.unique(source_xr)
    classes = [gt.classNameDict.get(u) for u in uniques]
    class_names = [g.short for g in classes]

    protocol = gt.AnnotationProtocol(class_names=class_names)

    crop_attrs = gt.AnnotationCropAttrs(
        name=crop_name,
        description=None,
        protocol=protocol)
    
    pre, post, _ = split_by_suffix(dest, ('.zarr',))
    crop_group: zarr.Group = access_zarr(
        pre,
        post, 
        attrs={'cellmap' : {'annotation': crop_attrs.dict()}},
        mode='w-')
    
    for un in uniques:
        class_name = gt.classNameDict.get(un).short
        data_unique = np.array((source_xr == un).astype('uint8'))
        num_present = int(data_unique.sum())
        num_absent = data_unique.size - num_present
        hist = {'absent': num_absent}

        label_name = class_name.lower().replace(' ', '_')
        label_group_path = os.path.join(crop_group.path, label_name)

        annotation_group_attrs = gt.MultiscaleGroupAttrs(
            class_name=class_name,
            description='',
            created_by=[],
            created_with=[],
            start_date=None,
            end_date=None,
            duration_days=None,
            annotation_type=annotation_type)
        
        annotation_array_attrs = gt.AnnotationArrayAttrs(
            class_name=class_name,
            histogram=hist,
            annotation_type=annotation_type
        )

        label_group = access_zarr(crop_group.store, 
                                 path=label_group_path,
                                 attrs={'cellmap': {'annotation': annotation_group_attrs.dict()},
                                        'multiscales': [ome_meta.dict()]}, mode='w-')
        
        label_array = access_zarr(crop_group.store, 
                                 path=os.path.join(label_group.path, array_name),
                                 shape=source_xr.shape,
                                 dtype=data_unique.dtype,
                                 compressor=Blosc(cname='zstd'),
                                 chunks=out_chunks,
                                 attrs={'cellmap': {'annotation' : annotation_array_attrs.dict()}},
                                 mode='w-')
        
        label_array[:] = data_unique

    return crop_group

@click.command
@click.argument("source", type=click.STRING)
@click.argument("dest", type=click.STRING)
@click.option("--name", type=click.STRING)
def cli(source, dest, name):
    split_annotations(source, dest, name)


