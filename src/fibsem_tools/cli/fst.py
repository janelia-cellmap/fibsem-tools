import click
from fibsem_tools.cli.db import db
from fibsem_tools.cli.label import label
from fibsem_tools.cli.array_copy import copy_group_cli, copy_array_cli
from fibsem_tools.cli.save_single_chunked import save_single_chunked_cli


@click.group("fst")
def fst():
    pass


fst.add_command(db)
fst.add_command(label)
fst.add_command(copy_array_cli)
fst.add_command(copy_group_cli)
fst.add_command(save_single_chunked_cli)