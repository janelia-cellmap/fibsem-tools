import os
from typing import Dict, Literal, Tuple, Union
import click
from xarray import DataArray
from fibsem_tools import read_xarray
import numpy as np
from fibsem_tools.io.airtable import (
    ImageInsert,
    LabelledAnnotationState,
    get_airbase,
    select_annotation_by_name,
    select_image_by_collection_and_name,
    upsert_images_by_location,
)
from fibsem_tools.io.zarr import get_store, get_url
import zarr
from zarr.storage import contains_array, contains_group
from fibsem_tools.io.util import split_by_suffix
from xarray_ome_ngff import get_adapters
from zarr.errors import ContainsArrayError, ContainsGroupError
from cellmap_schemas.annotation import (
    SemanticSegmentation,
)
from xarray_multiscale import multiscale, windowed_mode
from fibsem_tools.io.tif import access as access_tif
from fibsem_tools.metadata.split_annotation import create_spec

ome_adapters = get_adapters("0.4")
out_chunks = (256,) * 3

annotation_type = SemanticSegmentation(encoding={"absent": 0, "present": 1})


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


def split_annotations(
    source: str,
    dest: str,
    collection_name: str,
    image_name: str,
    class_encoding: Dict[str, LabelledAnnotationState],
    chunks: Union[Literal["auto"], Tuple[Tuple[int, ...], ...]] = "auto",
    compressor="auto",
    overwrite: bool = False,
) -> zarr.Group:
    assert image_name != ""
    if chunks == "auto":
        out_chunks = (64, 64, 64)
    else:
        out_chunks = chunks

    pre, post, _ = split_by_suffix(dest, (".zarr",))
    # fail fast if there's already a group there

    store = get_store(pre)

    if contains_group(store, post) and not overwrite:
        raise ContainsGroupError(f"{store.path}/{post}")

    if contains_array(store, post) and not overwrite:
        raise ContainsArrayError(f"{store.path}/{post}")

    source_fmt = guess_format(source)
    if source_fmt == "tif":
        _data = access_tif(source, memmap=False)
        img_db = select_image_by_collection_and_name(f"{collection_name}/{image_name}")
        coords = img_db.to_stt().to_coords(shape=_data.shape)
        data = DataArray(_data, coords=coords)

    multi = {
        m.name: m for m in multiscale(data, windowed_mode, (2, 2, 2), chunks=out_chunks)
    }

    spec = create_spec(
        data=multi,
        crop_name=image_name,
        class_encoding=class_encoding,
        compressor=compressor,
    )

    crop_group = spec.to_zarr(
        zarr.open(pre, mode="a").store, path=post, overwrite=overwrite
    )

    # write out all labels
    print(f"Saving all labels to {get_url(crop_group)}/all")
    for array_name, array in multi.items():
        array_path = os.path.join(crop_group.path, "all", array_name)
        arr_dest = zarr.Array(
            store=crop_group.store, path=array_path, write_empty_chunks=False
        )
        arr_dest[:] = array.data

    for class_name, value in class_encoding.items():
        print(f"Saving class {class_name} to {get_url(crop_group)}/{class_name}")
        for array_name, array in multi.items():
            array_path = os.path.join(crop_group.path, class_name, array_name)
            data_unique = np.array((array == value.numeric_label).astype("uint8"))
            arr_dest = zarr.Array(
                store=crop_group.store,
                path=array_path,
                write_empty_chunks=False,
            )
            arr_dest[:] = data_unique

    return crop_group


def update_airtable(
    collection_id: str, annotation_id: str, image_name: str, group: zarr.Group
):
    images_upsert = []
    base = get_airbase()
    classes_query_result = base.table("class").all()
    classes_dict = {
        res["fields"]["field_name"]: res["id"] for res in classes_query_result
    }
    for name, value in group.items():
        value_url = get_url(value)
        # remove 'file://' from urls until ML pipelines can handle it
        location = value_url.replace("file://", "")
        xr: DataArray = read_xarray(value_url)["s0"].data

        if name in classes_dict:
            class_name = [classes_dict[name]]
        elif name == "all":
            class_name = list(classes_dict.values())
        # upsert this image record
        images_upsert.append(
            ImageInsert.from_xarray(
                xr,
                name=f"{image_name}/{name}",
                collection=collection_id,
                location=location,
                value_type="label",
                image_type="human_segmentation",
                format="zarr",
                annotations=[annotation_id],
                clas=class_name,
            )
        )
    image_upsert_result = upsert_images_by_location(images_upsert)
    print(
        "Inserted {0} images, and updated {1}".format(
            len(image_upsert_result["createdRecords"]),
            len(image_upsert_result["updatedRecords"]),
        )
    )
    # update the annotation record
    # query the table again because it may have changed prior to saving the zarr arrays
    annotation_query_result_2 = select_annotation_by_name(
        image_name, resolve_images=False
    )
    new_image_ids = list(
        set(
            annotation_query_result_2.images
            + [x["id"] for x in image_upsert_result["records"]]
        )
    )
    base.table("annotation").update(
        annotation_query_result_2.id, fields={"images": new_image_ids}
    )


@click.command
@click.argument("source", type=click.STRING)
@click.argument("dest", type=click.STRING)
@click.argument("collection_name", type=click.STRING)
@click.argument("image_name", type=click.STRING)
@click.option("--overwrite", is_flag=True, default=False, type=click.BOOL)
@click.option("--no-update-db", is_flag=True, default=False, type=click.BOOL)
def crop_to_zarr(
    source,
    dest,
    collection_name: str,
    image_name: str,
    overwrite: bool,
    no_update_db: bool,
):
    """
    Convert a dense annotation stored as a TIFF file located at SOURCE into an OME-NGFF Zarr group located at DEST.

    This function requires access to the Cellmap annotation database.
    Set the AIRTABLE_API_KEY environment variable to the correct API key.
    You may also set that variable in a .env file local to the execution of this script.
    """

    annotation_query_result = select_annotation_by_name(image_name, resolve_images=True)
    collection_id = annotation_query_result.images[0].collection

    # only write out labels that are annotated
    class_encoding_annotated = {
        k: v
        for k, v in annotation_query_result.annotation_states().items()
        if v.annotated
    }

    result_group = split_annotations(
        source=source,
        dest=dest,
        image_name=image_name,
        collection_name=collection_name,
        class_encoding=class_encoding_annotated,
        overwrite=overwrite,
    )

    if not no_update_db:
        update_airtable(
            collection_id=collection_id,
            annotation_id=annotation_query_result.id,
            image_name=image_name,
            group=result_group,
        )
    """    
        if not no_update_db:
        images_upsert = []
        base = get_airbase()
        classes_query_result = base.table("class").all()
        classes_dict = {
            res["fields"]["field_name"]: res["id"] for res in classes_query_result
        }
        for name, value in result_group.items():
            value_url = get_url(value)
            xr: DataArray = read_xarray(value_url)["s0"].data

            if name in classes_dict:
                class_name = [classes_dict[name]]
            elif name == "all":
                class_name = list(classes_dict.values())
            # upsert this image record
            images_upsert.append(
                ImageInsert.from_xarray(
                    xr,
                    name=f"{image_name}/{name}",
                    collection=collection_id,
                    location=value_url,
                    value_type="label",
                    image_type="human_segmentation",
                    format="zarr",
                    annotations=[annotation_query_result.id],
                    clas=class_name,
                )
            )
        image_upsert_result = upsert_images_by_location(images_upsert)
        print(
            "Inserted {0} images, and updated {1}".format(
                len(image_upsert_result["createdRecords"]),
                len(image_upsert_result["updatedRecords"]),
            )
        )
        # update the annotation record
        # query the table again because it may have changed prior to saving the zarr arrays
        annotation_query_result_2 = select_annotation_by_name(
            image_name, resolve_images=False
        )
        new_image_ids = list(
            set(
                annotation_query_result_2.images
                + [x["id"] for x in image_upsert_result["records"]]
            )
        )
        base.table("annotation").update(
            annotation_query_result_2.id, fields={"images": new_image_ids}
        ) """
