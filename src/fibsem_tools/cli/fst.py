import click

from fibsem_tools.cli.array_copy import copy_array_cli, copy_group_cli
from fibsem_tools.cli.save_single_chunked import save_single_chunked_cli


@click.group("fst")
def fst():
    pass


fst.add_command(copy_array_cli)
fst.add_command(copy_group_cli)
fst.add_command(save_single_chunked_cli, "save-single-chunked")
