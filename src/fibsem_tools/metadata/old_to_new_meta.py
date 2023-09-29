import os
from typing import Dict, Optional, Tuple
import click
from pydantic import BaseModel
from xarray import DataArray
from fibsem_tools import read_xarray
import numpy as np
from fibsem_tools.io.zarr import get_store
from pydantic_zarr import GroupSpec, ArraySpec
import zarr
from zarr.storage import contains_array, contains_group
from fibsem_tools.io.util import split_by_suffix
from fibsem_tools.metadata.transform import STTransform
from numcodecs import Blosc
from xarray_ome_ngff import get_adapters
from zarr.errors import ContainsArrayError, ContainsGroupError
from cellmap_schemas.annotation import (
    SemanticSegmentation,
    AnnotationArrayAttrs,
    AnnotationGroupAttrs,
    CropGroupAttrs,
)
from dotenv import load_dotenv
load_dotenv()
ome_adapters = get_adapters("0.4")
out_chunks = (256,) * 3

annotation_type = SemanticSegmentation(encoding={"absent": 0, "present": 1})


class ImageRow(BaseModel):
    id: str
    size_x_pix: int
    size_y_pix: int
    size_z_pix: int
    resolution_x_nm: float
    resolution_y_nm: float
    resolution_z_nm: float
    offset_x_nm: float
    offset_y_nm: float
    offset_z_nm: float
    value_type: str
    image_type: str
    title: str

    def to_stt(self):
        return STTransform(
            order="C",
            units=("nm", "nm", "nm"),
            axes=("z", "y", "x"),
            scale=(self.resolution_z_nm, self.resolution_y_nm, self.resolution_x_nm),
            translate=(self.offset_z_nm, self.offset_y_nm, self.offset_x_nm),
        )


def get_airbase():
    from pyairtable import Api
    return Api(os.environ["AIRTABLE_API_KEY"]).base(os.environ["AIRTABLE_BASE_ID"])

class AnnotationState(BaseModel):
    present: Optional[bool]
    annotated: Optional[bool]
    sparse: Optional[bool]


def class_encoding_from_airtable(image_name: str):
    airbase = get_airbase()
    annotation_table = airbase.table('annotation')
    annotation_types = airbase.table('annotation_type').all(fields=['name','present', 'annotated', 'sparse'])
    annotation_types_parsed = {a['id']: AnnotationState(**{**{'present': False, 'annotated': False, 'sparse': False}, **a['fields']}) for a in annotation_types}
    result = annotation_table.first(
        formula='{image} = "' + image_name + '"')
    if result is None:
        raise ValueError(f'Airtable does not contain a record named {image_name}')
    else:
        fields = result['fields']
        out = fields.copy()
        for key in tuple(fields.keys()):
            try:
                value_str, *rest = key.split('_')
                value = int(value_str)
                # raises if the prefix is not an int
                annotation_type = annotation_types_parsed[fields[key][0]]
                if annotation_type.annotated:
                    out['_'.join(rest)] = int(value)
            except ValueError:
                # the field name was not of the form <number>_<class>
                pass
    return out       

def image_from_airtable(image_name: str) -> ImageRow:
    airbase = get_airbase()
    result = airbase.table("image").first(formula='{name} = "' + image_name + '"')
    if result is None:
        raise ValueError(f"Airtable does not contain a record named {image_name}")
    else:
        fields = result["fields"]
        try:
            return ImageRow(**fields, id=result["id"])
        except KeyError as e:
            raise ValueError(f"Missing field in airtable: {e}")


def coords_from_airtable(image_name: str, shape: Tuple[int, ...]) -> STTransform:
    return image_from_airtable(image_name).to_stt().to_coords(shape=shape)


def create_spec(
    data: DataArray, crop_name: str, array_name: str, class_encoding: Dict[str, int]
):

    ome_meta = ome_adapters.multiscale_metadata([data], [array_name])

    annotation_group_specs: dict[str, GroupSpec] = {}

    for class_name, value in class_encoding.items():
        data_unique = (data == value).astype("uint8")
        num_present = int(data_unique.sum())
        num_absent = data_unique.size - num_present
        hist = {"absent": num_absent}

        annotation_group_attrs = AnnotationGroupAttrs(
            class_name=class_name, description="", annotation_type=annotation_type
        )

        annotation_array_attrs = AnnotationArrayAttrs(
            class_name=class_name, histogram=hist, annotation_type=annotation_type
        )
        label_array_spec = ArraySpec.from_array(
            data_unique,
            chunks=out_chunks,
            compressor=Blosc(cname="zstd"),
            attrs={"cellmap": {"annotation": annotation_array_attrs}},
        )

        annotation_group_specs[class_name] = GroupSpec(
            attrs={
                "cellmap": {"annotation": annotation_group_attrs},
                "multiscales": [ome_meta.dict()],
            },
            members={array_name: label_array_spec},
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
        attrs={"cellmap": {"annotation": {crop_attrs}}},
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
            f"Could not figure out what format the file at {path} is using."
        )


def split_annotations(
    source: str, dest: str, crop_name: str, class_encoding: Dict[str, int]
) -> zarr.Group:

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
        coords = coords_from_airtable(crop_name, _data.shape)
        data = DataArray(_data, coords=coords)
    else:
        data = read_xarray(source)

    # todo: don't assume the biggest array is named s0!
    array_name = "s0"

    spec = create_spec(
        data=data,
        crop_name=crop_name,
        array_name=array_name,
        class_encoding=class_encoding,
    )

    crop_group = spec.to_zarr(
        zarr.NestedDirectoryStore(pre), path=post, overwrite=False
    )

    for class_name, value in class_encoding.items():
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
    class_encoding = class_encoding_from_airtable(name)
    split_annotations(source, dest, name, class_encoding)

if __name__ == "__main__":
    cli()
