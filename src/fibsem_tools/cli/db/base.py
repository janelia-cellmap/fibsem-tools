import click

from fibsem_tools.cli.db.query_image import query_image
from fibsem_tools.cli.db.update_image import update_image


@click.group("db")
def db():
    pass


db.add_command(query_image)
db.add_command(update_image)
