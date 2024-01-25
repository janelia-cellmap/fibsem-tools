import click
from fibsem_tools.cli.db import db
from fibsem_tools.cli.label import label


@click.group("fst")
def fst():
    pass


fst.add_command(db)
fst.add_command(label)
