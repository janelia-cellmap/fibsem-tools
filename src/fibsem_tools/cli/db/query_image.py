import click
from fibsem_tools.io.airtable import select_image_by_collection_and_name


@click.command
@click.argument("image_identifier", type=click.STRING)
def query_image(
    image_identifier: str,
):
    query_result = select_image_by_collection_and_name(image_identifier)
    print(query_result.json(indent=2))
