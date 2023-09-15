from typing import Literal, Union
import click
import zarr
from fibsem_tools import access
from rich import print
from fibsem_tools.io.zarr import get_chunk_keys
from rich.progress import track
import time
from dataclasses import dataclass

ChunkState = Literal["valid", "missing", "invalid"]


@dataclass
class Missing:
    """
    This class represents a chunk that was missing.
    """

    variant = "missing"


@dataclass
class Invalid:
    """
    This class represents a chunk that raised an exception upon loading / decompression.
    """

    variant = "invalid"
    exception: BaseException


@dataclass
class Valid:
    """
    This class represents a chunk that was valid.
    """

    variant = "valid"


class ChunkSetResults(dict[ChunkState, dict[str, Union[Missing, Valid, Invalid]]]):
    pass


def check_zarray(array: zarr.Array) -> dict[str, Union[Missing, Invalid, Valid]]:
    """
    Check the state of each chunk of a zarr array. This function iterates over the
    chunks of an array, attempts to access each chunk, and records whether that chunk
    is valid (the chunk was fetched + decompressed without incident), invalid (
    an exception was raised when loading + decompressing the chunk) or missing (the
    chunk was not found in storage).

    Parameters
    ----------

    array: Zarr.Array
        The zarr array to check.

    Returns
    -------

    A dict with string keys, where each key is the location of a chunk in the key
    space of the store object associated with the array, and each value is either a
    Valid, Missing, or Invalid object.
    """
    ckeys = tuple(get_chunk_keys(array))
    results = {}
    for ckey in track(ckeys, description="Checking chunks..."):
        try:
            array._decode_chunk(array.store[ckey])
            results[ckey] = Valid()
        except OSError as e:
            results[ckey] = Invalid(exception=e)
        except KeyError:
            results[ckey] = Missing()

    return results


@click.command()
@click.argument("array_path", type=click.STRING)
@click.option(
    "--valid",
    is_flag=True,
    show_default=True,
    default=False,
    help="report valid chunks",
)
@click.option(
    "--missing",
    is_flag=True,
    show_default=True,
    default=False,
    help="report missing chunks",
)
@click.option(
    "--invalid",
    is_flag=True,
    show_default=True,
    default=False,
    help="report invalid chunks",
)
@click.option(
    "--delete-invalid",
    is_flag=True,
    show_default=True,
    default=False,
    help="delete invalid chunks",
)
def cli(
    array_path: str, valid: bool, missing: bool, invalid: bool, delete_invalid: bool
):
    """
    Checks the chunks of a zarr array, prints the results as JSON, and optionally
    deletes invalid chunks.

    Parameters
    ----------

    array_path: string
        The path to the array.

    valid: bool
        Whether to report valid chunks. Default is False, which results in no output if
        a chunk is valid.

    missing: bool
        Whether to report missing chunks. Default is False, which results in no output
        if a chunk is missing.

    invalid: bool
        Whether to report invalid chunks. Default is True. An invalid chunk is defined
        as one which raises an OSError upon loading + decompression. This definition may
        change to include more exception types, but the basic idea is that a chunk is
        invalid if it has been corrupted or cannot be read with the compressor as
        defined in the array metadata.

    delete_invalid: bool
        Whether to delete invalid chunks. Default is False.

    """
    start = time.time()
    array = access(array_path, mode="r")
    all_results = check_zarray(array)
    # categorize
    results_categorized: ChunkSetResults = {"valid": {}, "missing": {}, "invalid": {}}
    for key, value in all_results.items():
        results_categorized[value.variant][key] = value

    to_show = {}

    for flag, opt in zip((valid, missing, invalid), ("valid", "missing", "invalid")):
        if flag:
            to_show[opt] = results_categorized[opt]
    print(to_show)
    if delete_invalid:
        array_a = access(array_path, mode="a")
        num_invalid = len(results_categorized["invalid"])
        for res in track(
            results_categorized["invalid"],
            description=f"Deleting {num_invalid} invalid chunks...",
        ):
            del array_a.store[res]
    print(f"Completed after {time.time() - start}s")


if __name__ == "__main__":
    cli()
