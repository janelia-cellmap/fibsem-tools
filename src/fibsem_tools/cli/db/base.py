import click

from fibsem_tools.cli.db.image import query_image, add_image


@click.group("db")
def db():
    pass


db.add_command(query_image)
db.add_command(add_image)
