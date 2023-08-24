from typing import Union
import click
from pydantic_zarr import GroupSpec, ArraySpec
import zarr
from fibsem_tools import read
from fibsem_tools.io.util import split_by_suffix
from rich import print

def parse_zarr_path(path: str) -> Union[ArraySpec, GroupSpec]:
    pre, post, suffix = split_by_suffix(path, ('.n5', '.zarr'))
    obj = read(path)
    if isinstance(obj, zarr.Array):
        result = ArraySpec.from_zarr(obj)
    elif isinstance(obj, zarr.Group):
        result = GroupSpec.from_zarr(obj)
    else:
        raise ValueError(f'Got an unparseable object: {type(obj)}')
    return result


@click.command()
@click.argument('path', type=click.STRING)
def cli(path: str):
    result = parse_zarr_path(path)
    print(result.json(indent=2))

if __name__ == "__main__":
    cli()
