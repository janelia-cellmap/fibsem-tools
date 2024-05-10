from typing import Literal, Optional

import click
import dask
import zarr
from dask.diagnostics import ProgressBar
from pydantic_zarr import ArraySpec, GroupSpec
from zarr.errors import ContainsGroupError

import fibsem_tools.io.dask as fsd
from fibsem_tools import access, read
from fibsem_tools.cli.base import parse_chunks, parse_compressor
from fibsem_tools.io.zarr import copyable


@click.group()
def data():
    pass


def copy_array(
    *,
    source_url: str,
    dest_url: str,
    region: str,
    access_mode: Literal["w", "a"],
    chunks: Optional[str],
    compressor: Optional[str],
    compressor_opts: Optional[str],
    no_copy_attrs: bool,
):
    dest: zarr.Array
    source = read(source_url)

    chunks = parse_chunks(chunks)
    compressor_instance = parse_compressor(
        source.compressor, compressor, compressor_opts
    )

    chunk_mismatch = False
    compressor_mismatch = False

    if access_mode == "w":
        dest = access(
            dest_url,
            mode="w",
            shape=source.shape,
            chunks=chunks,
            compressor=compressor_instance,
            dtype=source.dtype,
        )
    elif access_mode == "a":
        dest = access(
            dest_url,
            mode="a",
            shape=source.shape,
            chunks=chunks,
            compressor=compressor_instance,
            dtype=source.dtype,
        )
        msg = ""
        if chunks is not None and dest.chunks != chunks:
            chunk_mismatch = True
            msg += f"You requested chunks {chunks}, but the destination array at {dest_url} has different chunks: {dest.chunks}. "
        if compressor_instance is not None and compressor_instance != dest.compressor:
            compressor_mismatch = True
            msg += f"You requested compressor {compressor_instance}, but the destination array at {dest_url} has a different compressor: {dest.compressor}. "
        if any((chunk_mismatch, compressor_mismatch)):
            msg += (
                "Change the properties of the destination array, "
                "or use the `--overwrite` flag (which will overwrite the target)"
            )
            raise ValueError(msg)

    if region != "all":
        # to make this work, we need to resolve a slice to a tuple of slices that are aligned to chunk boundaries
        raise NotImplementedError

    # region_slices = parse_region(dest.shape, region)
    return copy_array(source, dest, keep_attrs=not no_copy_attrs)


@data.command("copy-array")
@click.argument("source_url", type=click.STRING)
@click.argument("dest_url", type=click.STRING)
@click.option("--region", type=click.STRING, default="all")
@click.option("--access-mode", type=click.STRING, default="a")
@click.option("--chunks", type=click.STRING)
@click.option("--compressor", type=click.STRING, default="same")
@click.option("--compressor-opts", type=click.STRING)
@click.option("--no-copy-attrs", type=click.BOOL, default=False, is_flag=True)
def copy_array_cli(
    source_url: str,
    dest_url: str,
    region: str,
    access_mode: Literal["w", "a"],
    chunks: Optional[str],
    compressor: Optional[str],
    compressor_opts: Optional[str],
    no_copy_attrs: bool,
):
    copy_array(
        source_url=source_url,
        dest_url=dest_url,
        region=region,
        access_mode=access_mode,
        chunks=chunks,
        compressor=compressor,
        compressor_opts=compressor_opts,
        no_copy_attrs=no_copy_attrs,
    )


@data.command("copy-group")
@click.argument("source_url", type=click.STRING)
@click.argument("dest_url", type=click.STRING)
@click.option("--access-mode", type=click.STRING, default="a")
@click.option("--chunks", type=click.STRING)
@click.option("--compressor", type=click.STRING, default="same")
@click.option("--compressor-opts", type=click.STRING)
@click.option("--no-copy-attrs", type=click.BOOL, default=False, is_flag=True)
def copy_group_cli(
    *,
    source_url: str,
    dest_url: str,
    access_mode: Literal["w", "a"],
    chunks: Optional[str],
    compressor: Optional[str],
    compressor_opts: Optional[str],
    no_copy_attrs: bool,
):
    source = read(source_url)

    dest: zarr.Group

    if access_mode == "w":
        dest = access(dest_url, mode="w")
    else:
        dest = access(dest_url, mode="a")
    chunks = parse_chunks(chunks)
    compressor_instance = parse_compressor(
        None, compressor=compressor, compressor_opts=compressor_opts
    )

    source_spec = GroupSpec.from_zarr(source)
    new_spec_dict = source_spec.dict()
    new_members = {}
    for key, member in source_spec.members.items():
        update = {}
        if chunks is not None:
            update["chunks"] = chunks
        if compressor_instance is not None:
            update["compressor"] = compressor_instance

        new_members[key] = ArraySpec(**(member.dict() | update)).dict()
    new_spec_dict["members"] = new_members
    dest_spec = GroupSpec(**new_spec_dict)
    try:
        dest_group = dest_spec.to_zarr(
            dest.store, path=dest.path, overwrite=(access_mode == "w")
        )
    except ContainsGroupError:
        dest_group = zarr.open(dest.store, path=dest.path, mode="a")
        if not copyable(source_spec, GroupSpec.from_zarr(dest_group)):
            msg = (
                f"It is not safe to copy {source} to {dest_group}. "
                f"Ensure that all the arrays in {source} have a counterpart in {dest_group} "
                "with the same dtype and shape attributes"
            )
            raise ValueError(msg)
    print(f"Copying {source_url} to {dest_url}.")

    if not no_copy_attrs:
        dest_group.attrs.update(source.attrs.asdict())

    for key, value in source.items():
        if not no_copy_attrs:
            dest_group[key].attrs.update(value.attrs.asdict())
        print(f"{key}")
        with ProgressBar():
            copy_bag = fsd.copy_array(value, dest_group[key], keep_attrs=False)
            dask.compute(copy_bag)


if __name__ == "__main__":
    data()
