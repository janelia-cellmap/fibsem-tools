import click
from typing import List, Tuple, Any, NamedTuple
from functools import partial

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
                Elements of {value} were parsed to {result},
                  which has repeating values.
                  """
            )
        return result
    except ValueError:
        raise click.BadParameter(
            f"""
            Input must be a comma-separated list of unique strings. Got {value} instead.
            """
        )


class ArrayMoveOp(NamedTuple):
    source: str
    dest: str
    dims: List[str]
    units: List[str]
    scale: List[int]
    translate: List[int]
    chunks: List[int]


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
def array_move_op_cli(
    source: str,
    dest: str,
    dims: Tuple[str, ...],
    units: Tuple[str, ...],
    scale: Tuple[float, ...],
    translate: Tuple[float, ...],
    chunks: Tuple[int, ...],
) -> ArrayMoveOp:
    rank = len(dims)
    stretchables = (units, scale, translate, chunks)
    units, scale, translate, chunks = tuple(
        map(partial(stretch_tuple, length=rank), stretchables)
    )
    return ArrayMoveOp(
        source=source,
        dest=dest,
        dims=list(dims),
        units=list(units),
        scale=list(scale),
        translate=list(translate),
        chunks=list(chunks),
    )
