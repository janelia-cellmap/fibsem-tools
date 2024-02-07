from glob import glob
import os
from fibsem_tools.cli.label.crop_to_zarr import split_annotations, update_airtable
from fibsem_tools.io.airtable import select_annotation_by_name

from rich import print
from pathlib import Path
import shutil
from dask import delayed
from distributed import Client, LocalCluster
from toolz import partition_all


sub_crops = [
    "crop20",
    "crop21",
    "crop22",
    "crop25",
    "crop26",
    "crop80",
    "crop81",
    "crop82",
    "crop83",
    "crop84",
    "crop97",
    "crop98",
    "crop99",
    "crop106",
    "crop115",
    "crop186",
    "crop187",
    "crop188",
    "crop189",
    "crop190",
    "crop191",
    "crop192",
    "crop193",
    "crop195",
    "crop196",
    "crop197",
    "crop198",
    "crop199",
    "crop200",
    "crop201",
    "crop202",
    "crop203",
    "crop206",
    "crop208",
    "crop209",
    "crop210",
    "crop211",
    "crop212",
    "crop213",
    "crop214",
    "crop216",
    "crop217",
    "crop218",
    "crop219",
    "crop220",
    "crop222",
    "crop224",
    "crop225",
    "crop226",
    "crop227",
    "crop228",
    "crop318",
    "crop338",
    "crop375",
    "crop378",
    "crop379",
    "crop380",
    "crop381",
]


def fname_to_image_name(fname: str) -> str:
    return Path(fname).parts[-2]


def fname_to_dataset_name(fname: str) -> str:
    return Path(fname).parts[-3]


def get_output_path(fname: str) -> str:
    image_name = fname_to_image_name(fname)
    dataset_name = fname_to_dataset_name(fname)
    return os.path.join(
        "/nrs/cellmap/data/", dataset_name, "staging", "groundtruth.zarr", image_name
    )


def try_split_annotation_from_fname(fname, overwrite=False):
    result = "success"
    image_name = fname_to_image_name(fname)
    collection_name = fname_to_dataset_name(fname)
    try:
        annotation_query_result = select_annotation_by_name(
            image_name, resolve_images=True
        )
        collection_id = annotation_query_result.images[0].collection

        output_path = get_output_path(fname)

        if os.path.exists(output_path) and overwrite:
            shutil.rmtree(output_path)

        crop_name = fname_to_image_name(fname)
        class_encoding_annotated = {
            k: v
            for k, v in annotation_query_result.annotation_states().items()
            if v.annotated
        }
        try:
            result_group = split_annotations(
                source=fname,
                dest=output_path,
                image_name=crop_name,
                collection_name=collection_name,
                class_encoding=class_encoding_annotated,
            )

            update_airtable(
                collection_id=collection_id,
                annotation_id=annotation_query_result.id,
                image_name=image_name,
                group=result_group,
            )

        except KeyError as e:
            result = e
        except IndexError as e:
            result = e
    except ValueError as e:
        result = e

    return result


if __name__ == "__main__":
    cl = Client(LocalCluster(host="*"))
    print(cl.dashboard_link)
    fnames = glob(
        "/groups/cellmap/cellmap/annotations/amira/*/crop*/crop*_labels_convert.tif"
    )

    results = []
    # partitioning is useful to avoid exceeding the airtable API rate limit
    partition_size = 30
    fnames_partitioned = tuple(partition_all(partition_size, fnames))

    for idx, fnames_chunk in enumerate(fnames_partitioned):
        print(f"iteration {idx} / {len(fnames_partitioned)}")
        tasks = [
            delayed(try_split_annotation_from_fname)(fname, overwrite=True)
            for fname in fnames_chunk
        ]
        result_local = cl.compute(tasks, sync=True)
        for fn, r in zip(fnames_chunk, result_local):
            print(f"{fn}: {r}")
        results.extend(result_local)
    print(results)
