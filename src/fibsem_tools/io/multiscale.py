from __future__ import annotations
from typing import Any, Literal, Optional, Sequence, Tuple, Union, List

from xarray import DataArray

import zarr
from fibsem_tools.io.util import normalize_chunks
from fibsem_tools.metadata.cosem import (
    CosemMultiscaleGroupV1,
    CosemMultiscaleGroupV2,
)
from fibsem_tools.metadata.neuroglancer import (
    NeuroglancerN5Group,
)
from numcodecs.abc import Codec
from xarray_ome_ngff.registry import get_adapters
from pydantic_zarr import GroupSpec, ArraySpec


NGFF_DEFAULT_VERSION = "0.4"
multiscale_metadata_types = ["neuroglancer", "cellmap", "cosem", "ome-ngff"]


def multiscale_group(
    arrays: Sequence[DataArray],
    metadata_types: List[str],
    array_paths: Union[Sequence[str], Literal["auto"]] = "auto",
    chunks: Union[Tuple[Tuple[int, ...], ...], Literal["auto"]] = "auto",
    name: Optional[str] = None,
    **kwargs,
) -> GroupSpec:
    """
    Generate multiscale metadata of the desired flavor from a list of DataArrays

    Arguments
    ---------

    arrays : Sequence[DataArray]
        The arrays to store.
    metadata_types : List[str]
        The metadata flavor(s) to use.
    array_paths : Sequence[str]
        The path for each array in storage, relative to the parent group.
    chunks : Union[Tuple[Tuple[int, ...], ...], Literal["auto"]], default is "auto"
        The chunks for the arrays instances. Either an explicit collection of
        chunk sizes, one per array, or the string "auto". If `chunks` is "auto" and
        the `data` attribute of the arrays is chunked, then each stored array
        will inherit the chunks of the input arrays. If the `data` attribute
        is not chunked, then each stored array will have chunks equal to the shape of
        the input array.
    name : Optional[str]
        The name for the multiscale group. Only relevant for metadata flavors that
        support this field, e.g. ome-ngff

    Returns
    -------

    A GroupSpec instance representing the multiscale group

    """
    if array_paths == "auto":
        array_paths = [f"s{idx}" for idx in range(len(arrays))]
    _chunks = normalize_chunks(arrays, chunks)

    group_attrs = {}
    array_attrs = {path: {} for path in array_paths}

    if any(f.startswith("ome-ngff") for f in metadata_types) and any(
        f.startswith("cosem") for f in metadata_types
    ):
        msg = f"""
        You requested {metadata_types}, but ome-ngff metadata and cosem metadata are 
        incompatible. Use just ome-ngff metadata instead.
        """
        raise ValueError(msg)

    for flavor in metadata_types:
        flave, _, version = flavor.partition("@")

        if flave == "neuroglancer":
            g_spec = NeuroglancerN5Group.from_xarrays(arrays, chunks=_chunks, **kwargs)
            group_attrs.update(g_spec.attrs.dict())
        elif flave == "cosem":
            if version == "2":
                g_spec = CosemMultiscaleGroupV2.from_xarrays(
                    arrays, name=name, chunks=_chunks, **kwargs
                )
            else:
                g_spec = CosemMultiscaleGroupV1.from_xarrays(
                    arrays, name=name, chunks=_chunks, **kwargs
                )
            group_attrs.update(g_spec.attrs.dict())
            for key, value in g_spec.members.items():
                array_attrs[key].update(**value.attrs.dict())
        elif flave == "ome-ngff":
            if version == "":
                version = NGFF_DEFAULT_VERSION
            adapters = get_adapters(version)
            group_attrs["multiscales"] = [
                adapters.multiscale_metadata(
                    arrays, name="", array_paths=array_paths
                ).dict()
            ]
        else:
            msg = (
                "Multiscale metadata type {flavor} is unknown."
                f"Try one of {multiscale_metadata_types}"
            )
            raise ValueError(msg)

    members = {
        path: ArraySpec.from_array(arr, attrs=array_attrs[path], chunks=cnks, **kwargs)
        for arr, path, cnks in zip(arrays, array_paths, _chunks)
    }
    return GroupSpec(attrs=group_attrs, members=members)


def prepare_multiscale(
    dest_url: str,
    scratch_url: Optional[str],
    arrays: List[DataArray],
    array_names: List[str],
    access_mode: Literal["w", "w-", "a"],
    metadata_types: List[str],
    store_chunks: Tuple[int, ...],
    compressor: Codec,
) -> Tuple[zarr.Group, zarr.Group]:
    if scratch_url is not None:
        # prepare the temporary storage
        scratch_names = array_names[1:]
        scratch_multi = arrays[1:]

        scratch_group = multiscale_group(
            scratch_url,
            scratch_multi,
            scratch_names,
            chunks=None,
            metadata_types=metadata_types,
            group_mode="w",
            array_mode="w",
            compressor=None,
        )
    else:
        scratch_group = None

    # prepare final storage
    dest_group = multiscale_group(
        dest_url,
        arrays,
        array_names,
        chunks=store_chunks,
        metadata_types=metadata_types,
        group_mode=access_mode,
        array_mode=access_mode,
        compressor=compressor,
    )

    return (dest_group, scratch_group)


# TODO: make this more robust
def is_multiscale_group(node: Any) -> bool:
    if isinstance(node, zarr.Group):
        if (
            "multiscales" in node.attrs
            or "scales" in node.attrs
            or "scale_factors" in node.attrs
        ):
            return True

    return False
