from fibsem_tools.io.xr import stt_from_array
import typer
from typing import List, Literal, Tuple
from fibsem_tools.io.multiscale import multiscale_group
from fibsem_tools import access, read
from fibsem_tools.io.dask import autoscale_chunk_shape, store_blocks
from xarray import DataArray
import dask.array as da
from xarray_multiscale import multiscale
from xarray_multiscale import windowed_mean, windowed_mode
from dask.utils import memory_repr
import time
import numpy as np
from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Input, Static, Button, TextLog
from fibsem_tools.metadata.transform import STTransform
from pydantic import BaseModel


class MultiscaleUIArgs(BaseModel):
    transform: STTransform
    source_path: str = ""
    dest_path: str = ""


def listify_str(value: List[str]):
    return [x.strip() for x in value[0].split(",")]


def listify_float(value: List[str]):
    return list(map(float, listify_str(value)))


def listify_int(value: List[str]):
    return list(map(int, listify_str(value)))


def normalize_unit(unit: str):
    if unit == "nm":
        return "nanometer"
    else:
        raise ValueError(f"Unit {unit} not recognized")


def normalize_downsampler(downsampler: str, data: DataArray):
    if downsampler == "auto":
        if data.dtype in (np.int32, np.int64):
            return windowed_mode
        else:
            return windowed_mean
    elif downsampler == "mode":
        return windowed_mode
    elif downsampler == "mean":
        return windowed_mean
    else:
        raise ValueError(
            'Invalid argument. downsampler must be one of ("auto", "mode", "mean")'
        )


class ConverterApp(App):
    TITLE = "Convert tif to zarr"
    SUB_TITLE = ""
    CSS_PATH = "styles.css"

    multi_args = reactive(
        MultiscaleUIArgs(
            transform=STTransform(
                axes=("z", "y", "x"),
                scale=(1,) * 3,
                translate=(0,) * 3,
                units=("nm",) * 3,
            )
        )
    )

    transform_name_map = dict(scale="resolution", translate="offset", units="units")

    def compose(self) -> ComposeResult:
        # the corner element
        yield Static()

        for dim in self.multi_args.transform.axes:
            yield Static(dim.upper())

        for tform in self.transform_name_map:
            mapped = self.transform_name_map[tform]
            vals = self.multi_args.transform.dict()[tform]
            yield Static(mapped, id=f"{tform}_static")
            for idx, dim in enumerate(self.multi_args.transform.axes):
                yield Input(
                    name=f"transform/{tform}/{idx}",
                    value=str(vals[idx]),
                    placeholder=f"Enter the {dim} {mapped}",
                )

        yield Static("Source path")
        yield Input(placeholder="enter the path to the source image", id="source_path")
        yield Static("Dest path")
        yield Input(placeholder="enter the path to the target image", id="dest_path")
        yield Static()
        yield Button("Convert", id="convert_button", disabled=False)
        yield TextLog(id="tlog")

    def on_button_pressed(self, message: Button.Pressed):
        self.exit(self.multi_args)

    def on_input_changed(self, message: Input.Changed):
        tlog = self.query_one(TextLog)
        tlog.write(message.__dict__)

        if message.input.name.startswith("transform"):
            outer, field, index = message.input.name.split("/")

            old_args = self.multi_args.dict()

            old_args[outer][field][int(index)] = message.value
            self.multi_args = MultiscaleUIArgs(**old_args)


def cli(
    source: str,
    dest: str,
    axes: List[str] = typer.Option(["z, y, x"], callback=listify_str),
    units: List[str] = typer.Option(["m, m, m"], callback=listify_str),
    scale: List[str] = typer.Option(["1, 1, 1"], callback=listify_float),
    translate: List[str] = typer.Option(["0, 0, 0"], callback=listify_float),
    chunks: List[str] = typer.Option(["128, 128, 128"], callback=listify_int),
    downsampler: str = typer.Option("auto"),
):
    pass


def convert_data(
    source: str,
    dest: str,
    transform: STTransform,
    chunks: Tuple[int, ...],
    downsampler: Literal["mean", "mode"],
):
    source_array = read(source)
    nbytes = source_array.nbytes
    typer.echo(
        f"""
    Loading tif file with size {memory_repr(nbytes)}. 
    Be advised that this process requires approximately 
    {memory_repr(nbytes * 2)} of available memory.
    """
    )
    start = time.time()
    read_chunks = autoscale_chunk_shape(
        chunk_shape=chunks,
        array_shape=source_array.shape,
        size_limit="1GB",
        dtype=source_array.dtype,
    )

    data = da.from_array(np.array(source_array), chunks=read_chunks)
    typer.echo(f"""Done loading tif after {time.time() - start:0.2f} s""")
    source_xr = stt_from_array(
        data,
        dims=transform.axes,
        scales=transform.scale,
        translates=transform.translate,
        units=transform.units,
    )
    reducer = normalize_downsampler(downsampler, data)
    multi = multiscale(source_xr, reducer, (2,) * data.ndim, chunks=chunks)

    for idx, m in enumerate(multi):
        m.name = f"s{idx}"

    typer.echo("Begin preparing storage at {dest}")
    start = time.time()

    multi_group, multi_arrays = multiscale_group(
        dest,
        multi,
        array_paths=[m.name for m in multi],
        metadata_types=["ome-ngff@0.4"],
        chunks=chunks,
        group_mode="w",
        array_mode="w",
    )
    typer.echo(f"Done preparing storage after {time.time() - start:0.2f}s")

    storage = [
        store_blocks(s.data, access(d, write_empty_chunks=False, mode="a"))
        for s, d in zip(multi, multi_arrays)
    ]
    start = time.time()
    typer.echo("Begin storing arrays.")
    da.compute(storage)
    typer.echo(f"Done storing arrays after {time.time() - start:0.2f}s")


if __name__ == "__main__":
    app = ConverterApp()
    ui_args = app.run()
    typer.echo(ui_args)
