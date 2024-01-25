from fibsem_tools import read, access
import fibsem_tools.io.dask as fsd
import click
from typing import Literal, Optional, Tuple
from numcodecs.abc import Codec
import numcodecs
import zarr
import json
import dask
from pydantic_zarr import GroupSpec, ArraySpec
from zarr.errors import ContainsGroupError
from dask.diagnostics import ProgressBar


@click.group()
def data():
    pass


def copyable(source_group: GroupSpec, dest_group: GroupSpec, strict: bool = False):
    """
    Check whether a Zarr group modeled by a GroupSpec `source_group` can be copied into the Zarr group modeled by GroupSpec `dest_group`.
    This entails checking that every (key, value) pair in `source_group.members` has a copyable counterpart in `dest_group.members`.
    Arrays are copyable if their shape and dtype match. Groups are copyable if their members are copyable.
    In general copyability as defined here is not symmetric for groups, because a `source_group`
    may have fewer members than `dest_group`, but if each of the shared members are copyable,
    then the source is copyable to the dest, but not vice versa.
    """

    if strict:
        if set(source_group.members.keys) != set(dest_group.members.keys):
            return False
    else:
        # extra members are allowed in the destination group
        if not set(source_group.members.keys()).issubset(
            set(dest_group.members.keys())
        ):
            return False

    for key_source, key_dest in zip(
        source_group.members.keys(), dest_group.members.keys()
    ):
        value_source = source_group.members[key_source]
        value_dest = dest_group.members[key_dest]

        if type(value_source) != type(value_dest):
            return False
        if isinstance(value_source, ArraySpec):
            # shape and dtype are the two properties that must match for bulk copying to work.
            if value_source.shape != value_dest.shape:
                return False
            if value_source.dtype != value_dest.dtype:
                return False
        else:
            # recurse into subgroups
            return copyable(value_source, value_dest, strict=strict)
    return True


def parse_chunks(chunks: Optional[str]) -> Optional[Tuple[int, ...]]:
    if isinstance(chunks, str):
        parts = chunks.split(",")
        return tuple(map(int, parts))
    else:
        return chunks


def parse_compressor(
    source_compressor: Optional[Codec], compressor: str, compressor_opts: Optional[str]
) -> Optional[Codec]:
    if compressor == "same":
        if isinstance(compressor_opts, str):
            msg = f"You provided compressor options {compressor_opts} but `compressor` is set to `same`. This is an error."
            raise ValueError(msg)
        return source_compressor

    compressor_class = getattr(numcodecs, compressor)
    if compressor_opts is None:
        compressor_opts = {}
    compressor_opts_dict = json.loads(compressor_opts)
    compressor_instance = eval("compressor_class(**compressor_opts_dict)")
    return compressor_instance


def parse_region(shape: Tuple[int, ...], region_spec: str) -> Tuple[slice, ...]:
    """
    convert a string into a tuple of slices
    """
    if region_spec == "all":
        return tuple(slice(0, s) for s in shape)
    else:
        results = []
        # depth represents whether we are inside a tuple (0) or between tuples (1)
        depth = 1
        for element in region_spec:
            if element == "(":
                # opening paren: start a new collection
                results.append([""])
                if depth == 0:
                    msg = f"Extraneous parenthesis in {region_spec}"
                    raise ValueError(msg)
                depth = 0
            elif element == "," and depth == 0:
                # comma between parens:
                # switch to the next element in the collection
                results[-1].append("")
            elif element == "," and depth == 1:
                # comma between parenthesized values
                # do nothing here
                pass
            elif element == ")":
                # closing paren
                # switch back to the first element in the collection
                if depth == 0:
                    depth = 1
                else:
                    msg = f"Extraneous parenthesis in {region_spec}"
                    raise ValueError(msg)
            elif element == " ":
                # whitespace. do nothing
                pass
            else:
                results[-1][-1] += element

    results_int = tuple(map(lambda v: tuple(map(int, v)), results))
    all_2 = all(tuple(len(x) == 2 for x in results_int))
    if not all_2:
        raise ValueError(
            f"All of the elements of region must have length 2. Got {results_int}"
        )

    results_slice = tuple(map(lambda v: slice(*v), results_int))
    return results_slice


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
