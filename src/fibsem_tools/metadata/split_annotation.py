import os
from typing import Dict, Literal, Tuple, Union
import click
from xarray import DataArray
from fibsem_tools import read_xarray
import numpy as np
from fibsem_tools.io.airtable import (
    class_encoding_from_airtable_by_image,
    coords_from_airtable,
)
from fibsem_tools.io.zarr import get_store
from pydantic_zarr import GroupSpec, ArraySpec
import zarr
from zarr.storage import contains_array, contains_group
from fibsem_tools.io.util import split_by_suffix
from numcodecs import Blosc
from xarray_ome_ngff import get_adapters
from zarr.errors import ContainsArrayError, ContainsGroupError
from cellmap_schemas.annotation import (
    SemanticSegmentation,
    AnnotationArrayAttrs,
    AnnotationGroupAttrs,
    CropGroupAttrs,
    wrap_attributes,
)
from xarray_multiscale import multiscale, windowed_mode

ome_adapters = get_adapters("0.4")
out_chunks = (64,) * 3

annotation_type = SemanticSegmentation(encoding={"absent": 0, "present": 1})


def create_spec(
    data: dict[str, DataArray], crop_name: str, class_encoding: Dict[str, int]
):
    ome_meta = ome_adapters.multiscale_metadata(
        tuple(data.values()), tuple(data.keys())
    )

    annotation_group_specs: dict[str, GroupSpec] = {}

    for class_name, value in class_encoding.items():
        label_array_specs = {}
        for array_name, array_data in data.items():
            data_unique = (array_data == value).astype("uint8")
            num_present = int(data_unique.sum())
            num_absent = data_unique.size - num_present
            hist = {"absent": num_absent}

            annotation_group_attrs = AnnotationGroupAttrs(
                class_name=class_name, description="", annotation_type=annotation_type
            )

            annotation_array_attrs = AnnotationArrayAttrs(
                class_name=class_name, histogram=hist, annotation_type=annotation_type
            )
            label_array_specs[array_name] = ArraySpec.from_array(
                data_unique,
                chunks=out_chunks,
                compressor=Blosc(cname="zstd"),
                attrs=wrap_attributes(annotation_array_attrs).dict(),
            )

        annotation_group_specs[class_name] = GroupSpec(
            attrs={
                **wrap_attributes(annotation_group_attrs).dict(),
                "multiscales": [ome_meta.dict()],
            },
            members=label_array_specs,
        )

    crop_attrs = CropGroupAttrs(
        name=crop_name,
        description=None,
        created_by=[],
        created_with=[],
        start_date=None,
        end_date=None,
        duration_days=None,
        class_names=tuple(class_encoding.keys()),
    )

    crop_group_spec = GroupSpec(
        attrs=wrap_attributes(crop_attrs).dict(),
        members=annotation_group_specs,
    )

    return crop_group_spec


def guess_format(path: str):
    if path.endswith(".tiff") or path.endswith(".tif"):
        return "tif"
    elif ".zarr" in path:
        return "zarr"
    elif ".n5" in path:
        return "n5"
    else:
        raise ValueError(
            f"Could not figure out what format the file at {path} is using. ",
            "Failed to find tif, tiff, n5, and zarr extensions.",
        )


def split_annotations(
    source: str,
    dest: str,
    crop_name: str,
    class_encoding: Dict[str, int],
    chunks: Union[Literal["auto"], Tuple[Tuple[int, ...], ...]] = "auto",
) -> zarr.Group:
    if chunks == "auto":
        out_chunks = (64, 64, 64)
    else:
        out_chunks = chunks

    pre, post, _ = split_by_suffix(dest, (".zarr",))
    # fail fast if there's already a group there

    store = get_store(pre)
    if contains_group(store, post):
        raise ContainsGroupError(f"{store.path}/{post}")

    if contains_array(store, post):
        raise ContainsArrayError(f"{store.path}/{post}")
    source_fmt = guess_format(source)
    if source_fmt == "tif":
        from fibsem_tools.io.tif import access as access_tif

        _data = access_tif(source, memmap=False)
        coords = coords_from_airtable(crop_name, shape=_data.shape)
        data = DataArray(_data, coords=coords)
    else:
        data = read_xarray(source)

    multi = {
        m.name: m for m in multiscale(data, windowed_mode, (2, 2, 2), chunks=out_chunks)
    }

    spec = create_spec(
        data=multi,
        crop_name=crop_name,
        class_encoding=class_encoding,
    )

    crop_group = spec.to_zarr(
        zarr.NestedDirectoryStore(pre), path=post, overwrite=False
    )

    for class_name, value in class_encoding.items():
        for array_name, data in multi.items():
            data_unique = np.array((data == value).astype("uint8"))
            arr = zarr.Array(
                store=crop_group.store,
                path=os.path.join(crop_group.path, class_name, array_name),
                write_empty_chunks=False,
            )
            arr[:] = data_unique

    return crop_group


@click.command
@click.argument("source", type=click.STRING)
@click.argument("dest", type=click.STRING)
@click.argument("name", type=click.STRING)
def cli(source, dest, name):
    class_encoding = class_encoding_from_airtable_by_image(name)
    split_annotations(source, dest, name, class_encoding)


if __name__ == "__main__":
    cli()
