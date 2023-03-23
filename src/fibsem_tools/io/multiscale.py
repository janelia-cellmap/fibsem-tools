from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union, List

from xarray import DataArray

import zarr
from fibsem_tools.io.core import AccessMode, create_group
from fibsem_tools.metadata.cosem import COSEMGroupMetadata
from fibsem_tools.metadata.neuroglancer import NeuroglancerN5GroupMetadata
from fibsem_tools.metadata.transform import STTransform
from zarr.errors import ContainsGroupError
from numcodecs.abc import Codec
from xarray_ome_ngff.registry import get_adapters


from fibsem_tools.io.util import Attrs, JSON

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
                raise ValueError(f"All values in chunks must be ints. Got {chunks}")
        except TypeError as e:
            raise e

    assert len(result) == len(arrays)
    assert tuple(map(len, result)) == tuple(
        x.ndim for x in arrays
    ), "Number of chunks per array does not equal rank of arrays"
    return result


def multiscale_metadata(
    arrays: Sequence[DataArray],
    metadata_types: List[str],
    array_paths: Optional[List[str]] = None,
) -> Tuple[Dict[str, JSON], List[Dict[str, JSON]]]:
    """
    Generate multiscale metadata of the desired flavor from a list of DataArrays

    Returns
    -------

    A tuple of dicts with string keys and JSON-serializable values

    """
    group_attrs = {}
    array_attrs: List[Dict[str, Any]] = [{}] * len(arrays)
    for flavor in set(metadata_types):
        if flavor == "neuroglancer":
            g_meta = NeuroglancerN5GroupMetadata.fromDataArrays(arrays)
            group_attrs.update(g_meta.dict())
        elif flavor == "cellmap" or flavor == "cosem":
            g_meta = COSEMGroupMetadata.fromDataArrays(arrays)
            group_attrs.update(g_meta.dict())
            for idx in range(len(array_attrs)):
                array_attrs[idx] = {
                    **STTransform.fromDataArray(arrays[idx]).dict(),
                    **array_attrs[idx],
                }
        elif flavor.startswith("ome-ngff"):
            if "@" in flavor:
                ngff_version = flavor.split("@")[-1]
            else:
                ngff_version = NGFF_DEFAULT_VERSION
            adapters = get_adapters(ngff_version)
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
    return group_attrs, array_attrs


def multiscale_group(
    group_url: str,
    arrays: List[DataArray],
    array_paths: List[str],
    chunks: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...], None],
    metadata_types: List[str],
    group_mode: AccessMode = "w-",
    array_mode: AccessMode = "w-",
    group_attrs: Optional[Attrs] = None,
    array_attrs: Optional[Sequence[Attrs]] = None,
    **kwargs: Any,
) -> Tuple[str, Tuple[str, ...]]:
    if array_attrs is None:
        array_attrs = [{}] * len(arrays)
    if group_attrs is None:
        group_attrs = {}

    mgroup_attrs, marray_attrs = multiscale_metadata(
        arrays, metadata_types, array_paths=array_paths
    )
    group_attrs.update(mgroup_attrs)
    [a.update(marray_attrs[idx]) for idx, a in enumerate(array_attrs)]

    _chunks = _normalize_chunks(arrays, chunks)
    try:
        paths = create_group(
            group_url,
            arrays,
            array_paths=array_paths,
            chunks=_chunks,
            group_attrs=group_attrs,
            array_attrs=array_attrs,
            group_mode=group_mode,
            array_mode=array_mode,
            **kwargs,
        )
        return paths
    except ContainsGroupError:
        raise FileExistsError(
            f"""
            The resource at {group_url} resolves to an existing group. Use 'w' or 'a' 
            access modes to enable writable / appendable access to this group.
            """
        )


def prepare_multiscale(
    dest_url: str,
    scratch_url: Optional[str],
    arrays: List[DataArray],
    array_names: List[str],
    access_mode: Literal["w", "w-", "a"],
    metadata_types: List[str],
    store_chunks: Tuple[int, ...],
    compressor: Codec,
) -> Tuple[str, str]:

    if scratch_url is not None:
        # prepare the temporary storage
        scratch_names = array_names[1:]
        scratch_multi = arrays[1:]

        scratch_group_url, scratch_array_urls = multiscale_group(
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
        scratch_array_urls = []

    # prepare final storage
    dest_group_url, dest_array_urls = multiscale_group(
        dest_url,
        arrays,
        array_names,
        chunks=store_chunks,
        metadata_types=metadata_types,
        group_mode=access_mode,
        array_mode=access_mode,
        compressor=compressor,
    )

    return dest_array_urls, scratch_array_urls


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
