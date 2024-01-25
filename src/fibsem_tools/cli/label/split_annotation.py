from glob import glob
import os
from fibsem_tools.metadata.split_annotation import (
    class_encoding_by_image,
    split_annotations,
)
from rich import print
from pathlib import Path
import shutil
from dask import delayed
from distributed import Client, LocalCluster
from toolz import partition_all


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
        class_encoding = class_encoding_by_image(image_name)
        output_path = get_output_path(fname)
        if os.path.exists(output_path) and overwrite:
            shutil.rmtree(output_path)
        crop_name = fname_to_image_name(fname)
        try:
            split_annotations(
                source=fname,
                dest=output_path,
                crop_name=crop_name,
                collection_name=collection_name,
                class_encoding=class_encoding,
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
