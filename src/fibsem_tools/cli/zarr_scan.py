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
    variant = "missing"


@dataclass
class Invalid:
    variant = "invalid"
    exception: BaseException


@dataclass
class Valid:
    variant = "valid"


class ChunkSetResults(dict[ChunkState, dict[str, Union[Missing, Valid, Invalid]]]):
    pass


def check_zarray(array: zarr.Array) -> dict[str, Union[Missing, Invalid, Valid]]:
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
def cli(array_path, valid, missing, invalid, delete_invalid):
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
