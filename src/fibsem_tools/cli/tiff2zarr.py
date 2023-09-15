from fibsem_tools.io.xr import stt_from_array
from fibsem_tools.io.util import JSON
from typing import List, Dict, Tuple, Any, Sequence, Literal
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
from textual.widgets import Input, Static, Button, TextLog, Footer, Header
from fibsem_tools.metadata.transform import STTransform
from pydantic import (
    BaseModel,
    ValidationError,
    FilePath,
    conlist,
)
from pathlib import Path
import os
import click
from functools import partial


class ArrayMoveWeak(BaseModel):
    source: str
    dest: str
    dims: conlist(str, unique_items=True)
    units: List[str]
    scale: List[float]
    translate: List[float]
    chunks: List[int]


class ArrayMoveStrict(BaseModel):
    source: FilePath
    dest: Path
    dims: conlist(str, unique_items=True)
    units: List[str]
    scale: List[float]
    translate: List[float]
    chunks: List[int]


def represents_int(s: Any) -> bool:
    """
    Returns `True` if the value can be parsed as an int, and `False` otherwise.
    """
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
    """
    Set an item in a nested JSON-compatible dict using a composite key (like a URL).
    """
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


class Tiff2Zarr(App):
    TITLE = "Convert tiff to zarr"
    SUB_TITLE = ""
    CSS_PATH = "tiff2zarr.css"
    BINDINGS = [("ctrl + c", "", "quit")]

    is_valid: bool = reactive(True)
    is_converting: bool = reactive(False)
    separator = "/"

    data: ArrayMoveStrict = reactive(None)

    output_model = ArrayMoveStrict
    transform_name_map = dict(scale="resolution", translate="offset", units="units")

    def __init__(self, data):
        super().__init__()
        self.data = data

    def compose(self) -> ComposeResult:
        yield Header()
        # the corner element
        yield Static("dims", classes="label")

        for dim in self.data["dims"]:
            yield Static(dim)

        for tform in self.transform_name_map:
            mapped = self.transform_name_map[tform]
            vals = self.data[tform]

            yield Static(mapped, id=f"{tform}_static", classes="label")
            for idx, dim in enumerate(self.data["dims"]):
                yield Input(
                    id=self.separator.join(map(str, (tform, idx))),
                    value=str(vals[idx]),
                    placeholder=f"Enter the {dim} {mapped}",
                )
        yield Static("Tiff path", classes="label")
        yield Input(
            placeholder="enter the path to the source image",
            id="source",
            value=self.data["source"],
        )
        yield Static("Zarr path", classes="label")
        yield Input(
            placeholder="enter the path to the directory for the zarr image",
            id="dest",
            value=self.data["dest"],
        )
        yield Static()
        yield Button("Convert", id="convert_button", disabled=not self.is_valid)
        yield TextLog(id="tlog", markup=True)
        yield Footer()

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
            for error in e.errors():
                field = error["loc"]
                msg = error["msg"]
                # uncomment this when we figure out how to enable it in dev mode
                tlog.write(f"Validation error at {field}: {msg}")

            for widget in self.query("Input"):
                if widget.id in paths:
                    widget.remove_class("valid")
                    widget.add_class("invalid")
                else:
                    widget.remove_class("invalid")
                    widget.add_class("valid")

    def on_button_pressed(self, message: Button.Pressed):
        tlog = self.query_one("#tlog")
        args: ArrayMoveStrict = self.output_model(**self.data)
        fname = Path(args.source).stem

        transform = STTransform(
            axes=args.dims, units=args.units, translate=args.translate, scale=args.scale
        )
        self.is_converting = True
        tlog.write("Begin converting data...")

        convert_data(
            source=args.source,
            dest=os.path.join(args.dest, fname + ".zarr/"),
            transform=transform,
            chunks=args.chunks,
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

    multiscale_group(
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
        store_blocks(
            s, access(os.path.join(dest, s.name), write_empty_chunks=False, mode="a")
        )
        for s in multi
    ]
    start = time.time()
    logger.write("Begin storing arrays.")
    da.compute(storage)
    logger.write(f"Done storing arrays after {time.time() - start:0.2f}s")


# this functionality should be accomplished by a padding function rather than
# a resizing function. oh well.
def stretch_tuple(x: Tuple[Any, ...], length: int) -> Tuple[Any, ...]:
    """
    'stretch' a tuple to reach a certain length by duplicating the last element.
    """
    if length < 1:
        raise ValueError(
            f"""
        The second argument to this function must be an integer greater than 0. 
        Got {length} instead.
        """
        )
    if len(x) == length:
        return x
    elif (residual := (length - len(x))) > 1:
        return x + (x[-1],) * residual
    else:
        raise ValueError(
            f"""
            Too many elements in `x`. Got {len(x)}, expected at most {length}
        """
        )


def normalize_csl(csl, typ):
    return tuple(map(typ, map(str.strip, csl.split(","))))


def normalize_csl_int(ctx, param, value):
    try:
        return normalize_csl(value, int)
    except ValueError:
        raise click.BadParameter("input must be a comma-separated list of integers.")


def normalize_csl_float(ctx, param, value):
    try:
        return normalize_csl(value, float)
    except ValueError:
        raise click.BadParameter("input must be a comma-separated list of numbers.")


def normalize_csl_str(ctx, param, value):
    return normalize_csl(value, str)


def normalize_csl_str_unique(ctx, param, value):
    try:
        result = normalize_csl(value, str)
        if not len(set(result)) == len(result):
            raise ValueError(
                f"""
                Elements of {value} were parsed to {result}, which has repeating values.
                """
            )
        return result
    except ValueError:
        raise click.BadParameter(
            f"""
            Input must be a comma-separated list of unique strings. Got {value} instead.
            """
        )


@click.command()
@click.option(
    "--scale",
    default="1",
    help="the distance, in world units, between samples. should be a comma-separated list",  # noqa
    type=click.UNPROCESSED,
    callback=normalize_csl_float,
)
@click.option(
    "--translate",
    default="0",
    help="the position, in world units, of the first sample. should be a comma-separated list",  # noqa
    type=click.UNPROCESSED,
    callback=normalize_csl_float,
)
@click.option(
    "--units",
    default="nm",
    help="the units of the coordinate grid. should be a comma-separated list",
    type=click.UNPROCESSED,
    callback=normalize_csl_str,
)
@click.option(
    "--dims",
    default="z,y,x",
    help="the names of the dimensions of the data. should be a comma-separated list",
    type=click.UNPROCESSED,
    callback=normalize_csl_str_unique,
)
@click.option(
    "--chunks",
    default="64",
    help="the chunk size of the output. should be a comma-separated list",
    type=click.UNPROCESSED,
    callback=normalize_csl_int,
)
@click.option(
    "--source", help="the path / url to the source data", default="", type=click.STRING
)
@click.option(
    "--dest",
    help="the path / url to the destination data",
    default="",
    type=click.STRING,
)
def cli(
    source: str,
    dest: str,
    dims: Tuple[str, ...],
    units: Tuple[str, ...],
    scale: Tuple[float, ...],
    translate: Tuple[float, ...],
    chunks: Tuple[int, ...],
) -> ArrayMoveWeak:
    rank = len(dims)
    stretchables = (units, scale, translate, chunks)
    units, scale, translate, chunks = tuple(
        map(list, map(partial(stretch_tuple, length=rank), stretchables))
    )
    move_op = ArrayMoveWeak(
        source=source,
        dest=dest,
        dims=dims,
        units=units,
        scale=scale,
        translate=translate,
        chunks=chunks,
    )
    return move_op


def guess_dest(source: str) -> str:
    first, suffix, last = source.partition(".tif")
    if suffix != "":
        return first + ".zarr" + last
    else:
        return ""


def run():
    move_op = cli(standalone_mode=False)
    data = move_op.dict()
    if move_op.dest == "" and move_op.source != "":
        data["dest"] = guess_dest(move_op.source)
    app = Tiff2Zarr(data=data)
    result = app.run()
    return result
