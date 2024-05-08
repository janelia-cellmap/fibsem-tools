from __future__ import annotations
from typing import Literal, Tuple, Union

from numcodecs.abc import Codec
from zarr.storage import BaseStore

from fibsem_tools.io.util import normalize_chunks
from fibsem_tools.metadata.cosem import multiscale_group as cosem_multiscale_group
from fibsem_tools.metadata.neuroglancer import (
    multiscale_group as neuroglancer_multiscale_group,
)

from pydantic_zarr.v2 import GroupSpec
from xarray import DataArray
from xarray_ome_ngff.v04.multiscale import model_group as ome_ngff_v04_multiscale_group
import zarr

NGFF_DEFAULT_VERSION = "0.4"
multiscale_metadata_types = ["neuroglancer", "cosem", "ome-ngff"]


def model_multiscale_group(
    arrays: dict[str, DataArray],
    metadata_type: Literal["neuroglancer_n5", "ome-ngff", "ome-ngff", "cosem"],
    chunks: Union[Tuple[Tuple[int, ...], ...], Literal["auto"]] = "auto",
    **kwargs,
) -> GroupSpec:
    """
    Generate a model of a multiscale group from a list of DataArrays

    Arguments
    ---------

    arrays : dict[str, DataArray]
        The arrays to store.
    metadata_type : Literal["neuroglancer_n5", "ome-ngff", "cosem"],
        The metadata flavor to use.
    chunks : Union[Tuple[Tuple[int, ...], ...], Literal["auto"]], default is "auto"
        The chunks for the arrays instances. Either an explicit collection of
        chunk sizes, one per array, or the string "auto". If `chunks` is "auto" and
        the `data` attribute of the arrays is chunked, then each stored array
        will inherit the chunks of the input arrays. If the `data` attribute
        is not chunked, then each stored array will have chunks equal to the shape of
        the input array.

    Returns
    -------

    A GroupSpec instance representing the multiscale group

    """
    _chunks = normalize_chunks(arrays.values(), chunks=chunks)

    if metadata_type == "neuroglancer":
        return neuroglancer_multiscale_group(arrays=arrays, chunks=_chunks, **kwargs)
    elif metadata_type == "cosem":
        return cosem_multiscale_group(arrays=arrays, chunks=_chunks, **kwargs)
    elif metadata_type.startswith("ome-ngff"):
        _, _, ome_ngff_version = metadata_type.partition("@")
        if ome_ngff_version in ("", "0.4"):
            return ome_ngff_v04_multiscale_group(
                arrays=arrays, transform_precision=5, chunks=_chunks, **kwargs
            )
        else:
            msg = (
                f"Metadata type {metadata_type} refers to an unsupported version of "
                "ome-ngff ({ome_ngff_version})"
            )
            raise ValueError(msg)

    else:
        msg = (
            f"Multiscale metadata type {metadata_type} is unknown."
            f"Try one of {multiscale_metadata_types}"
        )
        raise ValueError(msg)


def create_multiscale_group(
    *,
    store: BaseStore,
    path: str,
    arrays: dict[str, DataArray],
    metadata_type: Literal["neuroglancer", "cosem", "ome-ngff", "ome-ngff@0.4"],
    chunks: Union[Tuple[Tuple[int, ...], ...], Literal["auto"]] = "auto",
    compressor: Codec | Literal["auto"] = "auto",
    **kwargs,
) -> zarr.Group:
    group_model = model_multiscale_group(
        arrays=arrays, metadata_type=metadata_type, chunks=chunks, compressor=compressor
    )

    return group_model.to_zarr(store=store, path=path, **kwargs)
