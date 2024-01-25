import click

from fibsem_tools.cli.label.crop_to_zarr import crop_to_zarr


@click.group("label")
def label():
    pass


label.add_command(crop_to_zarr)
