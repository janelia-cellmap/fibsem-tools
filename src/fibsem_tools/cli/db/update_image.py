from fibsem_tools import read_xarray
from fibsem_tools.io.airtable import get_airbase
import click
from rich import print
import numpy as np
import xarray
from datatree import DataTree


def get_scale(xr: xarray.DataArray):
    return {
        dim: float(np.abs(xr.coords[dim][1] - xr.coords[dim][0])) for dim in xr.dims
    }


@click.command
@click.argument("collection_name", type=click.STRING)
@click.argument("image_name", type=click.STRING)
@click.option("--dry", is_flag=True, default=False)
def update_image(collection_name, image_name: str, dry: bool):
    airbase = get_airbase()
    formula = (
        'AND({name} = "' + image_name + '", {collection} = "' + collection_name + '")'
    )
    image_rows = airbase.table("image").all(formula=formula)
    if len(image_rows) > 1:
        raise ValueError("Too many images returned by the query.")
    if len(image_rows) == 0:
        msg = f"No images returned by the query. Please check that an image named {image_name} is associated with a collection called {collection_name}."
        raise ValueError(msg)

    img_row = image_rows[0]
    img_fields = img_row["fields"]
    try:
        location = img_fields["location"]
    except KeyError as e:
        msg = f"No location field found in the record. Here are the fields returned from airtable: {img_fields}"
        raise ValueError(msg) from e
    xr = read_xarray(location)
    if isinstance(xr, DataTree):
        # sort the DataArrays in this tree by size, and take the largest
        xr = sorted(xr.values(), key=lambda v: v.data.shape, reverse=True)[0].data

    scales = get_scale(xr)
    new_fields = {
        "size_x_pix": xr.shape[xr.dims.index("x")],
        "size_y_pix": xr.shape[xr.dims.index("y")],
        "size_z_pix": xr.shape[xr.dims.index("z")],
        "resolution_x_nm": scales["x"],
        "resolution_y_nm": scales["y"],
        "resolution_z_nm": scales["z"],
        "offset_x_nm": float(xr.coords["x"][0]),
        "offset_y_nm": float(xr.coords["y"][0]),
        "offset_z_nm": float(xr.coords["z"][0]),
    }
    print("Inferred spatial data:")
    print(new_fields)
    if not dry:
        result = airbase.table("image").update(
            record_id=img_row["id"], fields=new_fields
        )
        print(result)
    else:
        print("Doing nothing, because `dry` was set to `True`")
