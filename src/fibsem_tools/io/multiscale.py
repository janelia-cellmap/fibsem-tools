from __future__ import annotations
from typing import Any, Literal, Optional, Sequence, Tuple, Union, List

from xarray import DataArray

import zarr
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


def _normalize_chunks(
    arrays: Sequence[DataArray],
    chunks: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...], None],
) -> Tuple[Tuple[int, ...], ...]:
    if chunks is None:
        result: Tuple[Tuple[int, ...]] = tuple(v.data.chunksize for v in arrays)
    elif all(isinstance(c, tuple) for c in chunks):
        result = chunks
    else:
        try:
            all_ints = all((isinstance(c, int) for c in chunks))
            if all_ints:
                result = (chunks,) * len(arrays)
            else:
                msg = f"All values in chunks must be ints. Got {chunks}"
                raise ValueError(msg)
        except TypeError as e:
            raise e

    assert len(result) == len(arrays)
    assert tuple(map(len, result)) == tuple(
        x.ndim for x in arrays
    ), "Number of chunks per array does not equal rank of arrays"
    return result


def multiscale_group(
    arrays: Sequence[DataArray],
    metadata_types: List[str],
    array_paths: Union[List[str], Literal["auto"]] = "auto",
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
    name : Optional[str]
        The name for the multiscale group. Only relevant for metadata flavors that
        support this field, e.g. ome-ngff

    Returns
    -------

    A GroupSpec instance representing the multiscale group

    """
    if array_paths == "auto":
        array_paths = [f"s{idx}" for idx in range(len(arrays))]
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
            g_spec = NeuroglancerN5Group.from_xarrays(arrays, **kwargs)
            group_attrs.update(g_spec.attrs.dict())
        elif flave == "cosem":
            if version == "2":
                g_spec = CosemMultiscaleGroupV2.from_xarrays(
                    arrays, name=name, **kwargs
                )
            else:
                g_spec = CosemMultiscaleGroupV1.from_xarrays(
                    arrays, name=name, **kwargs
                )
            group_attrs.update(g_spec.attrs.dict())

            for key, value in g_spec.items.items():
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
            raise ValueError(
                f"""
                Multiscale metadata type {flavor} is unknown. Try one of 
                {multiscale_metadata_types}
                """
            )
    members = {
        path: ArraySpec.from_array(arr, attrs=array_attrs[path], **kwargs)
        for arr, path in zip(arrays, array_paths)
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
