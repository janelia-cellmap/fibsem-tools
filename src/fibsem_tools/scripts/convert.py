from fibsem_tools.io.xr import stt_from_array
from fibsem_tools.io.types import JSON
from typing import List, Literal, Tuple, Dict, Sequence
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
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Input, Static, Button, TextLog
from fibsem_tools.metadata.transform import STTransform
from pydantic import (
    BaseModel,
    DirectoryPath,
    PositiveFloat,
    ValidationError,
    FilePath,
    conlist,
)
from pathlib import Path
import os


def represents_int(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True


def ensure_key(data, key):
    if isinstance(data, Sequence) and represents_int(key):
        return int(key)
    else:
        return key


def url_getitem(data: Dict[str, JSON], url: str, separator: str):
    first, *rest = url.split(separator)

    first = ensure_key(data, first)

    if len(rest) == 0:
        return data[first]
    else:
        subdata = data[first]
        return url_getitem(subdata, separator.join(rest), separator)


def url_setitem(data: Dict[str, JSON], url: str, value: JSON, separator: str):
    parts = url.split(separator)

    if len(parts) == 1:
        last = ensure_key(data, parts[0])
        data[last] = value
        return
    else:
        url_setitem(data[parts[0]], separator.join(parts[1:]), value, separator)


class Params(BaseModel):
    axes: conlist(str, unique_items=True)
    units: List[str]
    scale: List[PositiveFloat]
    translate: List[float]
    source_path: FilePath
    dest_path: DirectoryPath


class MultiscaleArgs(BaseModel):
    transform: STTransform
    source_path: FilePath
    dest_path: str


def validation_error_to_paths(e: ValidationError, separator: str):
    return (separator.join(map(str, err["loc"])) for err in e.errors())


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
    is_valid: bool = reactive(True)
    is_converting: bool = reactive(False)
    separator = "/"
    chunks = (128, 128, 128)
    data = reactive(
        dict(
            axes=["z", "y", "x"],
            scale=[4] * 3,
            translate=[0] * 3,
            units=["nm"] * 3,
            source_path="",
            dest_path="",
        )
    )

    output_model = Params
    transform_name_map = dict(scale="resolution", translate="offset", units="units")

    def compose(self) -> ComposeResult:
        # the corner element
        yield Static(classes="label")

        for dim in self.data["axes"]:
            yield Static(dim.upper())

        for tform in self.transform_name_map:
            mapped = self.transform_name_map[tform]
            vals = self.data[tform]

            yield Static(mapped, id=f"{tform}_static", classes="label")
            for idx, dim in enumerate(self.data["axes"]):
                yield Input(
                    id=self.separator.join(map(str, (tform, idx))),
                    value=str(vals[idx]),
                    placeholder=f"Enter the {dim} {mapped}",
                )

        yield Static("Source path", classes="label")
        yield Input(placeholder="enter the path to the source image", id="source_path")
        yield Static("Dest path", classes="label")
        yield Input(placeholder="enter the path to the target image", id="dest_path")
        yield Static()
        yield Button("Convert", id="convert_button", disabled=not self.is_valid)
        yield TextLog(id="tlog", markup=True)

    def watch_is_converting(self, is_converting):
        for widget in self.query("Input"):
            widget.disabled = is_converting
        self.query_one("#convert_button").disabled = is_converting

    def watch_is_valid(self, is_valid: bool):
        if is_valid:
            for widget in self.query("Input"):
                widget.remove_class("invalid")
                widget.add_class("valid")

        try:
            self.query_one("#convert_button").disabled = not is_valid
        except NoMatches:
            pass

    def on_input_changed(self, message: Input.Changed):
        tlog = self.query_one("#tlog")
        data = self.data.copy()
        url_setitem(data, message.input.id, message.value, self.separator)

        try:
            self.output_model(**data)
            self.data = data
            self.is_valid = True

        except ValidationError as e:
            paths = tuple(validation_error_to_paths(e, self.separator))
            self.is_valid = False
            tlog.write(e)

            for widget in self.query("Input"):
                if widget.id in paths:
                    widget.remove_class("valid")
                    widget.add_class("invalid")
                else:
                    widget.remove_class("invalid")
                    widget.add_class("valid")

    def on_button_pressed(self, message: Button.Pressed):
        tlog = self.query_one("#tlog")
        args: Params = self.output_model(**self.data)
        fname = Path(args.source_path).stem

        transform = STTransform(
            axes=args.axes, units=args.units, translate=args.translate, scale=args.scale
        )
        self.is_converting = True
        self.log("hi")
        tlog.write("Begin converting data...")

        convert_data(
            source=args.source_path,
            dest=os.path.join(args.dest_path, fname + ".zarr/"),
            transform=transform,
            chunks=self.chunks,
            downsampler="mode",
            logger=tlog,
        )

        tlog.write("Done converting data.")
        self.is_converting = False


def convert_data(
    source: str,
    dest: str,
    transform: STTransform,
    chunks: Tuple[int, ...],
    downsampler: Literal["mean", "mode"],
    logger,
):
    source_array = read(source)
    nbytes = source_array.nbytes
    logger.write(
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
    logger.write(f"""Done loading tif after {time.time() - start:0.2f} s""")
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

    logger.write(f"Begin preparing storage at {dest}")
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
    logger.write(f"Done preparing storage after {time.time() - start:0.2f}s")

    storage = [
        store_blocks(s.data, access(d, write_empty_chunks=False, mode="a"))
        for s, d in zip(multi, multi_arrays)
    ]
    start = time.time()
    logger.write("Begin storing arrays.")
    da.compute(storage)
    logger.write(f"Done storing arrays after {time.time() - start:0.2f}s")


def run():
    app = ConverterApp()
    result = app.run()
    return result


if __name__ == "__main__":
    run()
