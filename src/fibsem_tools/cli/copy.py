import os
from typing import Any, Literal, Optional

import click
import dask
from numcodecs import Zstd
from rich import print
from xarray_multiscale import multiscale, windowed_mean

from fibsem_tools.cli.fst import fst
from fibsem_tools.io.core import access, read_xarray
from fibsem_tools.io.multiscale.multiscale import model_multiscale_group
from fibsem_tools.io.zarr import get_store, get_url, parse_url


@fst.command("convert")
@click.argument("source", type=click.STRING)
@click.argument("dest", type=click.STRING)
@click.option("--overwrite", is_flag=True)
@click.option("--write-empty-chunks", is_flag=True)
def convert_cli(source: str, dest: str, overwrite: bool, write_empty_chunks: bool):
    mrc_multiscale(
        source_path=source,
        dest_path=dest,
        overwrite=overwrite,
        write_empty_chunks=write_empty_chunks,
    )


def mrc_multiscale(
    source_path: str,
    dest_path: str,
    *,
    source_format: str = "auto",
    source_chunks: str = "auto",
    dest_format: str = "auto",
    dest_chunks: str = "auto",
    multiscale_options: Optional[dict[str, Any]] = {},
    compute: Literal["local", "dask-local", "dask-lsf"] = "local",
    overwrite: bool = False,
    write_empty_chunks: bool = True,
) -> None:
    """
    Copy data from source to dest
    """

    # todo: make these dimension agnostic
    scale_factors = (2, 2, 2)
    out_chunks = (64, 64, 64)

    if source_format == "auto":
        # guess the format
        source_xr = read_xarray(source_path)

    multi_plan = multiscale(source_xr, windowed_mean, scale_factors=scale_factors)

    array_paths = [f"s{idx}" for idx in range(len(multi_plan))]

    multi_group_spec = model_multiscale_group(
        multi_plan,
        ["ome-ngff"],
        compressor=Zstd(level=3),
        array_paths=array_paths,
        chunks=out_chunks,
    )

    store_path, node_path = parse_url(dest_path)

    # create the group
    _ = multi_group_spec.to_zarr(
        get_store(store_path), path=node_path, overwrite=overwrite
    )

    # create writable handles for all the arrays
    dest_arrays = [
        access(
            os.path.join(store_path, node_path, array_path),
            mode="a",
            write_empty_chunks=write_empty_chunks,
        )
        for array_path in array_paths
    ]

    if compute == "local":
        # load everything into memory and create numpy arrays, then save them
        print("Begin generating multiscale images.")
        multi_arrays_local = dask.compute(multi_plan)
        print("Done generating multiscale images.")
        print("Begin saving images.")
        for src, dst in zip(multi_arrays_local, dest_arrays):
            print(f"saving {src} to {get_url(dst)}")
            dst[:] = src
        print("Done saving images.")
    else:
        raise NotImplementedError(f"cannot handle compute mode {compute} at this time.")


def array_to_array(source, dest):
    """
    copy the contents of an array to another array
    """

    # iterate over the output chunks of the target array

    # output_chunks = chunk_keys(read(dest))
    raise NotImplementedError


def group_to_group():
    """
    Copy the contents of a group to another group
    """
    ...

    raise NotImplementedError


"""     if dest_format == 'auto':
        dest = access(dest)

    operation: Literal['array->array', 'array->multiscale group', 'group->group'] = 'array->array'
    
    if operation == 'array->array':
        # schedule slicing + saving
        dest[:] = source_xr
    elif operation == 'array->multiscale group':
        # schedule pyramid + save
        pass
    elif operation == 'group->group':
        # schedule slicing + saving for each array
 """
