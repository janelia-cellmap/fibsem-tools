import click
from fibsem_tools.cli.serve import serve_cli


@click.group("fst")
def fst():
    pass


fst.add_command(serve_cli, "serve")
