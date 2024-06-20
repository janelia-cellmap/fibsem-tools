from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    from numcodecs.abc import Codec

import json

import numcodecs


def parse_compressor(
    source_compressor: Codec | None, compressor: str, compressor_opts: str | None
) -> Codec | None:
    if compressor == "same":
        if isinstance(compressor_opts, str):
            msg = f"You provided compressor options {compressor_opts} but `compressor` is set to `same`. This is an error."
            raise ValueError(msg)
        return source_compressor

    compressor_class = getattr(numcodecs, compressor)
    if compressor_opts is None:
        compressor_opts = "{}"
    compressor_opts_dict = json.loads(compressor_opts)
    return compressor_class(**compressor_opts_dict)


def parse_chunks(chunks: str | None) -> tuple[int, ...] | None:
    if isinstance(chunks, str):
        parts = chunks.split(",")
        return tuple(map(int, parts))
    return chunks


def parse_region(shape: tuple[int, ...], region_spec: str) -> tuple[slice, ...]:
    """
    convert a string into a tuple of slices
    """
    if region_spec == "all":
        return tuple(slice(0, s) for s in shape)

    results = []
    # depth represents whether we are inside a tuple (0) or between tuples (1)
    depth = 1
    for element in region_spec:
        if element == "(":
            # opening paren: start a new collection
            results.append([""])
            if depth == 0:
                msg = f"Extraneous parenthesis in {region_spec}"
                raise ValueError(msg)
            depth = 0
        elif element == "," and depth == 0:
            # comma between parens:
            # switch to the next element in the collection
            results[-1].append("")
        elif element == "," and depth == 1:
            # comma between parenthesized values
            # do nothing here
            pass
        elif element == ")":
            # closing paren
            # switch back to the first element in the collection
            if depth == 0:
                depth = 1
            else:
                msg = f"Extraneous parenthesis in {region_spec}"
                raise ValueError(msg)
        elif element == " ":
            # whitespace. do nothing
            pass
        else:
            results[-1][-1] += element

    results_int = tuple(tuple(map(int, v)) for v in results)
    all_2 = all(tuple(len(x) == 2 for x in results_int))
    if not all_2:
        msg = f"All of the elements of region must have length 2. Got {results_int}"
        raise ValueError(msg)

    return tuple(slice(*v) for v in results_int)


def parse_content_type(data: str) -> Literal["scalar", "label"]:
    if data not in ("scalar", "label"):
        msg = f"Must be scalar or label, got {data}"
        raise ValueError(msg)
    return data
