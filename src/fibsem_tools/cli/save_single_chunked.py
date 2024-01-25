from typing import List, Literal

from xarray import DataArray
from fibsem_tools.io.airtable import select_image_by_collection_and_name
import os
from fibsem_tools import access
from xarray_multiscale import multiscale, windowed_mean, windowed_mode
from xarray_multiscale.multiscale import downsampling_depth
from time import time
import dask
from fibsem_tools.io.core import read_xarray

from fibsem_tools.io.multiscale import multiscale_group
from fibsem_tools.io.zarr import get_store, parse_url
import click
from rich import print
from dask.array.core import slices_from_chunks, normalize_chunks

store_chunks = (64, 64, 64)



# read this much of the image per iteration
input_chunk = (1024, -1, -1)


def get_location_airtable(image_name: str, dataset_name: str) -> str:
    images = select_image_by_collection_and_name(image_name, dataset_name)

    if len(images) > 1:
        raise ValueError(
            "Got more than 1 image when querying airtable. Make your query more specific."
        )
    if len(images) == 0:
        raise ValueError(
            f"No images returned with the query {image_name=} {dataset_name=}"
        )
    else:
        image = images[0]

    return image.location


def save_single_chunked(
    source_url: str, dest_url: str, content_type: Literal["scalar", "label"]
):
    source_template = read_xarray(source_url)
    scale_factors = (2,) * source_template.ndim
    depth = downsampling_depth(source_template.shape, scale_factors=scale_factors)
    if content_type == "scalar":
        reducer = windowed_mean
    elif content_type == "label":
        reducer = windowed_mode
    else:
        msg = f'Unrecognized value for `content_type`. Got {content_type}, expected one of ("scalar", "label")'
        raise ValueError(msg)
    multi_template = multiscale(source_template, reducer, scale_factors=scale_factors)[
        :depth
    ]

    array_names = [f"s{idx}" for idx in range(len(multi_template))]
    dest_groupspec = multiscale_group(
        multi_template, ["ome-ngff"], array_paths=array_names, chunks=store_chunks
    )

    store_path, group_path = parse_url(dest_url)
    dest_group = dest_groupspec.to_zarr(
        get_store(store_path), group_path, overwrite=True
    )
    slices = slices_from_chunks(normalize_chunks(input_chunk, source_template.shape))

    num_iter = len(slices)
    for idx, slce in enumerate(slices):
        slce_display = [[s.start, s.stop] for s in slce]
        print_prefix = f"iter {idx} / {num_iter}:"
        print(print_prefix)
        print(f"\tloading data from {slce_display}")
        start = time()
        source = source_template[slce].compute(scheduler="threads")

        print(f"\tloading data from {slce_display} took {time() - start}s")
        start = time()
        # this should be done by indexing the first multiscale collection, but i worry about dask issues
        multi = multiscale(source, reducer, scale_factors=scale_factors)
        multi_actual: List[DataArray]
        multi_actual, *_ = dask.compute(multi, scheduler="threads")
        print(f"\tDownsampling data from {slce_display} took {time() - start}s")

        start = time()
        for idx, val in enumerate(multi_actual):
            origin = (
                multi_template[idx].coords[dim].data.tolist().index(val.coords[dim][0])
                for dim in val.dims
            )
            selection = tuple(
                slice(ogn, ogn + sh) for ogn, sh in zip(origin, val.shape)
            )
            selection_display = [[s.start, s.stop] for s in selection]
            print(f"\tsaving level {idx} at {selection_display}")
            arr_out = access(os.path.join(dest_url, array_names[idx]), mode="a")
            arr_out[selection] = val.data
        print(f"\tsaving took {time() - start} s")


@click.command
@click.argument("image_name", type=click.STRING)
@click.argument("dataset_name", type=click.STRING)
@click.argument("dest_url", type=click.STRING)
def cli(image_name: str, dataset_name: str, dest_url: str):
    source_url = get_location_airtable(image_name, dataset_name)
    save_single_chunked(source_url, dest_url)


if __name__ == "__main__":
    cli()
