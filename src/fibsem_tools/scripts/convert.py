from fibsem_tools.io.xr import stt_from_array
import typer
from typing import List
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
from textual.widgets import Input, Static


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
    CSS_PATH = "styles.css"
    resolution = reactive([1, 1, 1])
    dims = ("z", "y", "x")
    transform_parameters = ("resolution", "offset", "unit")

    def compose(self) -> ComposeResult:
        # the corner element
        yield Static()

        for dim in self.dims:
            yield Static(dim.upper())

        for tform in self.transform_parameters:
            yield Static(tform, id=f"{tform}_static")
            for dim in self.dims:
                yield Input(
                    name=f"{tform}_{dim}", placeholder=f"Enter the {dim} {tform}"
                )


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
        data, dims=axes, scales=scale, translates=translate, units=units
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
    app.run()
    # typer.run(cli)
