from typing import Any, List, Literal, Optional, Tuple

from xarray import DataArray
from fibsem_tools.cli.base import parse_chunks, parse_compressor, parse_content_type
import os
from fibsem_tools import access
from xarray_multiscale import multiscale, windowed_mean, windowed_mode
from xarray_multiscale.multiscale import downsampling_depth
from time import time
import dask
from fibsem_tools.io.core import read_xarray

from fibsem_tools.io.multiscale import model_multiscale_group
from fibsem_tools.io.zarr import ensure_spec, parse_url
import click
from rich import print
from dask.array.core import slices_from_chunks, normalize_chunks
from numcodecs.abc import Codec


# todo: widen return type to Tuple[slice, ...]
def parse_start_index(data: Optional[Any]) -> int:
    if data is None:
        return 0
    return int(data)


def convert(
    source_url: str,
    dest_url: str,
    *,
    content_type: Literal["scalar", "label"],
    in_chunks: Tuple[int, ...],
    start_index: int,
    out_chunks: Tuple[int, ...],
    compressor: Codec,
):
    source_template = read_xarray(source_url)
    scale_factors = (2,) * source_template.ndim
    depth = downsampling_depth(source_template.shape, scale_factors=scale_factors)
    if content_type == "scalar":
        reducer = windowed_mean
    elif content_type == "label":
        reducer = windowed_mode

    multi_template = multiscale(source_template, reducer, scale_factors=scale_factors)[
        :depth
    ]

    array_names = [f"s{idx}" for idx in range(len(multi_template))]
    dest_groupspec = model_multiscale_group(
        multi_template,
        ["ome-ngff"],
        array_paths=array_names,
        chunks=out_chunks,
        compressor=compressor,
    )

    store_path, group_path = parse_url(dest_url)

    # dest_group = dest_groupspec.to_zarr(
    #    get_store(store_path), group_path, overwrite=False
    # )

    _ = ensure_spec(dest_groupspec, access(store_path, mode="a").store, group_path)
    slices = slices_from_chunks(normalize_chunks(in_chunks, source_template.shape))

    num_iter = len(slices)
    for idx, slce in tuple(enumerate(slices))[start_index:]:
        slce_display = [[s.start, s.stop] for s in slce]
        print_prefix = f"iter {idx} / {num_iter}:"
        print(print_prefix)
        print(f"\tloading data from {slce_display}")
        start = time()
        source = source_template[slce].compute(scheduler="threads")

        print(f"\tloading data from {slce_display} took {time() - start}s")
        start = time()
        # this should be done by indexing the first multiscale collection, but i worry that dask indexing problems
        # will give this bad performance
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
            # da.from_array(val.data, chunks=[k * 3 for k in out_chunks]).store(arr_out, region=selection, lock=None)
            arr_out[selection] = val.data
        print(f"\tsaving took {time() - start} s")


@click.command("convert")
@click.argument("source_url", type=click.STRING)
@click.argument("dest_url", type=click.STRING)
@click.argument("content_type", type=click.STRING)
@click.option("--in-chunks", type=click.STRING)
@click.option("--start-index", type=click.STRING)
@click.option("--out-chunks", type=click.STRING)
@click.option("--compressor", type=click.STRING, default="Zstd")
@click.option("--compressor-opts", type=click.STRING)
def save_single_chunked_cli(
    source_url: str,
    dest_url: str,
    content_type: str,
    in_chunks: Optional[str],
    start_index: Optional[int],
    out_chunks: Optional[str],
    compressor: Optional[str],
    compressor_opts: Optional[str],
):
    content_type_parsed = parse_content_type(content_type)

    if in_chunks is None:
        in_chunks_parsed = (1024, -1, -1)
    else:
        in_chunks_parsed = parse_chunks(in_chunks)

    if out_chunks is None:
        out_chunks_parsed = (64, 64, 64)
    else:
        out_chunks_parsed = parse_chunks(out_chunks)

    start_index_parsed = parse_start_index(start_index)

    compressor_instance = parse_compressor(
        None, compressor=compressor, compressor_opts=compressor_opts
    )
    # todo: allow specifying a region instead of using the stupid start_index hack
    convert(
        source_url,
        dest_url,
        content_type=content_type_parsed,
        in_chunks=in_chunks_parsed,
        start_index=start_index_parsed,
        out_chunks=out_chunks_parsed,
        compressor=compressor_instance,
    )
