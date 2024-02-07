from fibsem_tools import read_xarray
from typing import Optional
from fibsem_tools.io.airtable import ImageInsert, ImageUpdate, get_airbase
from pyairtable.formulas import match
from pyairtable import formulas
import click
from datatree import DataTree
from rich import print


@click.command("add-image")
@click.argument("collection_name", type=click.STRING)
@click.argument("image_name", type=click.STRING)
@click.argument("location", type=click.STRING)
@click.option("--id", type=click.STRING)
@click.option("--format", type=click.STRING)
@click.option("--image-type", type=click.STRING)
@click.option("--value-type", type=click.STRING)
@click.option("--update", is_flag=True, default=False)
def add_image(
    collection_name: str,
    image_name: str,
    location: str,
    id: Optional[str],
    format: Optional[str],
    image_type: Optional[str],
    value_type: Optional[str],
    update: bool,
):
    if id is not None and update is False:
        raise ValueError("Cannot specify an id with update set to False.")

    base = get_airbase()
    collection = base.table("collection").first(formula=f'id = "{collection_name}"')
    if collection is None:
        msg = f"A collection named {collection_name} could not be found. Add that collection to the database and try again."
        raise ValueError(msg)
    collection_id = collection["id"]
    img_table = base.table("image")
    if id is not None:
        precedent = img_table.get(id)
    else:
        formula = match({"location": location})
        precedent = img_table.first(formula=formula)

    xr = read_xarray(location)
    if isinstance(xr, DataTree):
        # todo: sort by size instead of relying on a name
        xr = xr["s0"].data
    if precedent is not None:
        print(f"There is already a record in airtable with location={location}.")
        print(precedent)

        if update is False:
            raise ValueError(
                "A pre-existing record was found, but update was set to `False`. To update that record, re-run this command with the `--update` flag."
            )
        else:
            precedent_collection = base.table("collection").get(
                precedent["fields"]["collection"][0]
            )

            if precedent_collection is not None:
                precedent_collection_name = precedent_collection["fields"]["id"]
                if precedent_collection_name != collection_name:
                    msg = f"The pre-existing record is associated with collection named {precedent_collection_name}, which differs from the `collection_name` provided to this program ({collection_name}). This program cannot be used to update the `collection_name` field. Alter the database and try again."
                    raise ValueError(msg)

            precedent_image_name = precedent["fields"]["name"]
            if precedent_image_name is not None and precedent_image_name != image_name:
                msg = f"The pre-existing record has name {precedent_image_name}, which differs from the `image_name` provided to this program ({image_name}). This program cannot be used to update the `image_name` field. Alter the database and try again."
                raise ValueError(msg)

            to_update = ImageUpdate.from_xarray(
                xr,
                id=precedent["id"],
                image_type=image_type,
                value_type=value_type,
                format=format,
            )
            update_result = to_update.to_airtable(img_table)
            print("Update result: ")
            print(update_result)
    else:
        to_insert = ImageInsert.from_xarray(
            xr,
            image_type=image_type,
            value_type=value_type,
            format=format,
            name=image_name,
            location=location,
            collection=[collection_id],
        ).to_airtable(table=img_table)
        print(to_insert)


@click.command("query-image")
@click.argument("collection_name", type=click.STRING)
@click.argument("image_name", type=click.STRING)
def query_image(collection_name: str, image_name):
    airbase = get_airbase()
    formula = formulas.match({"collection": collection_name, "name": image_name})
    query_result = airbase.table("image").all(formula=formula)
    print([x for x in query_result])
