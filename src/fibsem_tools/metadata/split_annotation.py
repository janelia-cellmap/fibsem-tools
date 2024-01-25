from typing import Dict
from xarray import DataArray
from fibsem_tools.io.airtable import LabelledAnnotationState
from pydantic_zarr import GroupSpec, ArraySpec
from numcodecs import Zstd
from xarray_ome_ngff import get_adapters
from cellmap_schemas.annotation import (
    SemanticSegmentation,
    AnnotationArrayAttrs,
    AnnotationGroupAttrs,
    CropGroupAttrs,
    wrap_attributes,
)
from numcodecs.abc import Codec

ome_adapters = get_adapters("0.4")
out_chunks = (64,) * 3
default_compressor = Zstd(level=3)
annotation_type = SemanticSegmentation(encoding={"absent": 0, "present": 1})


def create_spec(
    data: dict[str, DataArray],
    crop_name: str,
    class_encoding: Dict[str, LabelledAnnotationState],
    *,
    compressor: Codec = "auto",
) -> GroupSpec:
    ome_meta = ome_adapters.multiscale_metadata(
        tuple(data.values()), tuple(data.keys())
    )
    if compressor == "auto":
        compressor = default_compressor

    annotation_group_specs: dict[str, GroupSpec] = {}

    for class_name, value in class_encoding.items():
        label_array_specs = {}
        for array_name, array_data in data.items():
            data_unique = (array_data == value.numeric_label).astype("uint8")
            num_present = int(data_unique.sum())
            num_absent = data_unique.size - num_present
            hist = {"absent": num_absent}

            annotation_group_attrs = AnnotationGroupAttrs(
                class_name=class_name, description="", annotation_type=annotation_type
            )

            annotation_array_attrs = AnnotationArrayAttrs(
                class_name=class_name,
                complement_counts=hist,
                annotation_type=annotation_type,
            )
            label_array_specs[array_name] = ArraySpec.from_array(
                data_unique,
                chunks=out_chunks,
                compressor=compressor,
                attrs=wrap_attributes(annotation_array_attrs).dict(),
            )

        annotation_group_specs[class_name] = GroupSpec(
            attrs={
                **wrap_attributes(annotation_group_attrs).dict(),
                "multiscales": [ome_meta.dict()],
            },
            members=label_array_specs,
        )

    # insert all the labels as a vanilla ome-ngff group
    annotation_group_specs["all"] = GroupSpec(
        attrs={"multiscales": [ome_meta.dict()]},
        members={
            key: ArraySpec.from_array(value, compressor=compressor, chunks=out_chunks)
            for key, value in data.items()
        },
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
        msg = (
            f"Could not figure out what format the file at {path} is using. ",
            "Failed to find tif, tiff, n5, and zarr extensions.",
        )
        raise ValueError(msg)


""" 
def split_annotations(
    source: str,
    dest: str,
    image_identifier: str,
    class_encoding: Dict[str, int],
    chunks: Union[Literal["auto"], Tuple[Tuple[int, ...], ...]] = "auto",
) -> zarr.Group:
    if chunks == "auto":
        out_chunks = (64, 64, 64)
    else:
        out_chunks = chunks

    crop_name = image_identifier.split('/')[-1]
    assert crop_name  != ''
    pre, post, _ = split_by_suffix(dest, (".zarr",))
    # fail fast if there's already a group there

    store = get_store(pre)
    if contains_group(store, post):
        raise ContainsGroupError(f"{store.path}/{post}")

    if contains_array(store, post):
        raise ContainsArrayError(f"{store.path}/{post}")
    source_fmt = guess_format(source)
    if source_fmt == "tif":
        _data = access_tif(source, memmap=False)
        img_db = get_image_by_identifier(image_identifier)
        coords = img_db.to_stt().to_coords(shape=_data.shape)
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

    crop_group = spec.to_zarr(zarr.storage.FSStore(pre), path=post, overwrite=False)

    # write all labels to a zarr group

    for class_name, value in class_encoding.items():
        for array_name, data in multi.items():
            data_unique = np.array((data == value).astype("uint8"))
            arr = zarr.Array(
                store=crop_group.store,
                path=os.path.join(crop_group.path, class_name, array_name),
                write_empty_chunks=False,
            )
            arr[:] = data_unique

    return crop_group """

""" 
@click.command
@click.argument("source", type=click.STRING)
@click.argument("dest", type=click.STRING)
@click.argument("collection_name", type=click.STRING)
@click.argument("image_name", type=click.STRING)
def cli(source, dest, collection_name: str, image_name: str):
    class_encoding = class_encoding_by_image(image_name)
    split_annotations(
        source=source,
        dest=dest,
        crop_name=image_name,
        collection_name=collection_name,
        class_encoding=class_encoding,
    )


if __name__ == "__main__":
    cli()
 """
