import os
from typing import Optional
import click
from fibsem_tools import read_xarray
import numpy as np
import fibsem_tools.metadata.groundtruth as gt
from pydantic_zarr import GroupSpec, ArraySpec
from pathlib import Path
import zarr
from fibsem_tools.io.util import split_by_suffix
from numcodecs import Blosc
from xarray_ome_ngff import get_adapters

ome_adapters = get_adapters("0.4")
out_chunks = (256,) * 3

annotation_type = gt.SemanticSegmentation(encoding={"absent": 0, "present": 1})


def split_annotations(source: str, dest: str, name: Optional[str] = None) -> zarr.Group:
    tree = read_xarray(source)
    # todo: don't assume the biggest array is named s0!
    array_name = "s0"
    source_xr = tree[array_name].data

    ome_meta = ome_adapters.multiscale_metadata([source_xr], [array_name])
    if name is None:
        crop_name = Path(source).stem
    else:
        crop_name = name

    uniques = np.unique(source_xr)
    all_class_names = tuple(g.short for g in gt.classNameDict.values())
    observed_class_names = [gt.classNameDict.get(u).short for u in uniques]
    annotation_group_names = tuple(
        c.lower().replace(" ", "_") for c in observed_class_names
    )
    annotation_group_specs: dict[str, GroupSpec] = {}

    for idx, un in enumerate(uniques):
        class_name = gt.classNameDict.get(un).short
        data_unique = np.array((source_xr == un).astype("uint8"))
        num_present = int(data_unique.sum())
        num_absent = data_unique.size - num_present
        hist = {"absent": num_absent}

        annotation_group_attrs = gt.AnnotationGroupAttrs(
            class_name=class_name, description="", annotation_type=annotation_type
        )

        annotation_array_attrs = gt.AnnotationArrayAttrs(
            class_name=class_name, histogram=hist, annotation_type=annotation_type
        )
        label_array_spec = ArraySpec.from_array(
            data_unique,
            chunks=out_chunks,
            compressor=Blosc(cname="zstd"),
            attrs=gt.annotation_attrs_wrapper(annotation_array_attrs.dict()),
        )

        annotation_group_specs[annotation_group_names[idx]] = GroupSpec(
            attrs={
                **gt.annotation_attrs_wrapper(annotation_group_attrs.dict()),
                "multiscales": [ome_meta.dict()],
            },
            members={array_name: label_array_spec},
        )

    crop_attrs = gt.CropGroupAttrs(
        name=crop_name,
        description=None,
        created_by=[],
        created_with=[],
        start_date=None,
        end_date=None,
        duration_days=None,
        class_names=all_class_names,
        index=dict(zip(observed_class_names, annotation_group_specs)),
    )

    pre, post, _ = split_by_suffix(dest, (".zarr",))

    crop_group_spec = GroupSpec(
        attrs=gt.annotation_attrs_wrapper(crop_attrs.dict()),
        members=annotation_group_specs,
    )

    crop_group = crop_group_spec.to_zarr(
        zarr.NestedDirectoryStore(pre), path=post, overwrite=False
    )

    # save data inside arrays
    for idx, un in enumerate(uniques):
        data_unique = np.array((source_xr == un).astype("uint8"))
        arr = zarr.Array(
            store=crop_group.store,
            path=os.path.join(observed_class_names[idx], array_name),
            mode="w-",
            write_empty_chunks=False,
        )
        arr[:] = data_unique

    return crop_group


@click.command
@click.argument("source", type=click.STRING)
@click.argument("dest", type=click.STRING)
@click.option("--name", type=click.STRING)
def cli(source, dest, name):
    split_annotations(source, dest, name)
