import click
from fibsem_tools import read_xarray


@click.command
@click.argument("source", type=click.STRING)
def main(source: str) -> None:
    read_xarray(source)


if __name__ == "__main__":
    main()
