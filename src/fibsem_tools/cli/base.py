import json
from typing import Literal, Optional, Tuple

import numcodecs
from numcodecs.abc import Codec


def parse_compressor(
    source_compressor: Optional[Codec], compressor: str, compressor_opts: Optional[str]
) -> Optional[Codec]:
    if compressor == "same":
        if isinstance(compressor_opts, str):
            msg = f"You provided compressor options {compressor_opts} but `compressor` is set to `same`. This is an error."
            raise ValueError(msg)
        return source_compressor

    compressor_class = getattr(numcodecs, compressor)
    if compressor_opts is None:
        compressor_opts = "{}"
    compressor_opts_dict = json.loads(compressor_opts)
    compressor_instance = eval("compressor_class(**compressor_opts_dict)")
    return compressor_instance


def parse_chunks(chunks: Optional[str]) -> Optional[Tuple[int, ...]]:
    if isinstance(chunks, str):
        parts = chunks.split(",")
        return tuple(map(int, parts))
    else:
        return chunks


def parse_region(shape: Tuple[int, ...], region_spec: str) -> Tuple[slice, ...]:
    """
    convert a string into a tuple of slices
    """
    if region_spec == "all":
        return tuple(slice(0, s) for s in shape)
    else:
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

    results_int = tuple(map(lambda v: tuple(map(int, v)), results))
    all_2 = all(tuple(len(x) == 2 for x in results_int))
    if not all_2:
        raise ValueError(
            f"All of the elements of region must have length 2. Got {results_int}"
        )

    results_slice = tuple(map(lambda v: slice(*v), results_int))
    return results_slice


def parse_content_type(data: str) -> Literal["scalar", "label"]:
    if data not in ("scalar", "label"):
        msg = f"Must be scalar or label, got {data}"
        raise ValueError(msg)
    return data
