from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Union

import dask.array as da
import zarr
from pydantic import BaseModel
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from typing_extensions import deprecated
from xarray import DataArray
from xarray_ome_ngff import DaskArrayWrapper, ZarrArrayWrapper

from fibsem_tools.io.util import normalize_chunks
from fibsem_tools.metadata.transform import STTransform, stt_from_array, stt_to_coords


def normalize_paths(
    arrays: Sequence[DataArray], paths: Union[Sequence[str], Literal["auto"]]
):
    if paths == "auto":
        _paths = [f"s{idx}" for idx in range(len(arrays))]
    else:
        _paths = paths
    return _paths


class ScaleMetaV1(BaseModel):
    path: str
    transform: STTransform


class MultiscaleMetaV1(BaseModel):
    name: Optional[str]
    datasets: list[ScaleMetaV1]


class MultiscaleMetaV2(BaseModel):
    name: Optional[str]
    datasets: list[str]


@deprecated(
    "CosemGroupMetadataV1 is deprecated, use multiscale.cosem.GroupAttrs from the cellmap-schemas library instead"
)
class CosemGroupMetadataV1(BaseModel):
    """
    Multiscale metadata used by COSEM for multiscale datasets saved in N5/Zarr groups.
    """

    multiscales: list[MultiscaleMetaV1]

    @classmethod
    def from_xarrays(
        cls,
        arrays: dict[str, DataArray],
        name: Optional[str] = None,
    ):
        """
        Generate multiscale metadata from a sequence of DataArrays.

        Parameters
        ----------

        arrays : Sequence[xarray.DataArray]
            The collection of arrays from which to generate multiscale metadata. These
            arrays are assumed to share the same `dims` attributes, albeit with varying
            `coords`.
        name : Optional[str]
            The name for the multiresolution collection

        Returns
        -------
        COSEMGroupMetadataV1
        """

        multiscales = [
            MultiscaleMetaV1(
                name=name,
                datasets=[
                    ScaleMetaV1(path=path, transform=stt_from_array(array=arr))
                    for path, arr in arrays.items()
                ],
            )
        ]
        return cls(name=name, multiscales=multiscales)


@deprecated("CosemGroupMetadataV2 is deprecated. Do not use it.")
class CosemGroupMetadataV2(BaseModel):
    """
    Multiscale metadata used by COSEM for multiscale datasets saved in N5/Zarr groups.
    """

    multiscales: list[MultiscaleMetaV2]

    @classmethod
    def from_xarrays(
        cls,
        arrays: Sequence[DataArray],
        paths: Union[Sequence[str], Literal["auto"]] = "auto",
        name: Optional[str] = None,
    ):
        """
        Generate multiscale metadata from a sequence of DataArrays.

        Parameters
        ----------

        arrays : Sequence[xarray.DataArray]
            The collection of arrays from which to generate multiscale metadata. These
            arrays are assumed to share the same `dims` attributes, albeit with varying
            `coords`.
        paths: Union[Sequence[str], Literal["auto"]] = "auto"
            The names for each of the arrays in the multiscale
            collection. If set to "auto", arrays will be named automatically according
            to the scheme `s0` for the largest array, s1 for second largest, and so on.
        name : Optional[str], default is None.
            The name for the multiresolution collection.

        Returns
        -------
        COSEMGroupMetadataV2

        """

        _paths = normalize_paths(arrays, paths)

        multiscales = [
            MultiscaleMetaV2(
                name=name,
                datasets=_paths,
            )
        ]
        return cls(name=name, multiscales=multiscales, paths=_paths)


@deprecated(
    "CosemArrayAttrs is deprecated, use multiscale.cosem.ArrayAttrs from the cellmap-schemas library instead"
)
class CosemArrayAttrs(BaseModel):
    transform: STTransform


@deprecated(
    "CosemMultiscaleArray is deprecated, use multiscale.cosem.Array from the cellmap-schemas library instead"
)
class CosemMultiscaleArray(ArraySpec):
    attributes: CosemArrayAttrs

    @classmethod
    def from_xarray(cls, array: DataArray, **kwargs):
        attrs = CosemArrayAttrs(transform=stt_from_array(array))
        return cls.from_array(array, attributes=attrs, **kwargs)


@deprecated(
    "CosemMultiscaleGroupV1 is deprecated, use multiscale.cosem.Group from the cellmap-schemas library instead"
)
class CosemMultiscaleGroupV1(GroupSpec):
    attributes: CosemGroupMetadataV1
    members: dict[str, CosemMultiscaleArray]

    @classmethod
    def from_xarrays(
        cls,
        arrays: dict[str, DataArray],
        chunks: Union[tuple[tuple[int, ...], ...], Literal["auto"]] = "auto",
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Convert a collection of DataArray to a GroupSpec with CosemMultiscaleV1 metadata

        Parameters
        ----------

        arrays: dict[str, DataArray]
            The arrays comprising the multiscale image.
        chunks : Union[Tuple[Tuple[int, ...], ...], Literal["auto"]], default is "auto"
            The chunks for the `ArraySpec` instances. Either an explicit collection of
            chunk sizes, one per array, or the string "auto". If `chunks` is "auto" and
            the `data` attribute of the arrays is chunked, then each ArraySpec
            instance will inherit the chunks of the arrays. If the `data` attribute
            is not chunked, then each `ArraySpec` will have chunks equal to the shape of
            the source array.
        paths: Union[Sequence[str], Literal["auto"]] = "auto"
            The names for each of the arrays in the multiscale
            collection. If set to "auto", arrays will be named automatically according
            to the scheme `s0` for the largest array, s1 for second largest, and so on.
        name: Optional[str], default is None
            The name for the multiscale collection.
        **kwargs:
            Additional keyword arguments that will be passed to the `ArraySpec`
            constructor.
        """

        _chunks = normalize_chunks(arrays, chunks)

        attrs = CosemGroupMetadataV1.from_xarrays(arrays, name)

        array_specs = {
            key: CosemMultiscaleArray.from_xarray(arr, chunks=cnks, **kwargs)
            for arr, cnks, key in zip(arrays.values(), arrays.keys(), _chunks)
        }

        return cls(attributes=attrs, members=array_specs)


@deprecated("CosemGroupV2 is deprecated. Do not use it.")
class CosemMultiscaleGroupV2(GroupSpec):
    attributes: CosemGroupMetadataV2
    members: dict[str, CosemMultiscaleArray]

    @classmethod
    def from_xarrays(
        cls,
        arrays: Iterable[DataArray],
        chunks: Union[tuple[tuple[int, ...]], Literal["auto"]] = "auto",
        paths: Union[Sequence[str], Literal["auto"]] = "auto",
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Convert a collection of DataArray to a GroupSpec with CosemMultiscaleV2 metadata

        Parameters
        ----------

        arrays: Iterable[DataArray]
            The arrays comprising the multiscale image.
        chunks : Union[Tuple[Tuple[int, ...], ...], Literal["auto"]], default is "auto"
            The chunks for the `ArraySpec` instances. Either an explicit collection of
            chunk sizes, one per array, or the string "auto". If `chunks` is "auto" and
            the `data` attribute of the arrays is chunked, then each ArraySpec
            instance will inherit the chunks of the arrays. If the `data` attribute
            is not chunked, then each `ArraySpec` will have chunks equal to the shape of
            the source array.
        paths: Union[Sequence[str], Literal["auto"]] = "auto"
            The names for each of the arrays in the multiscale
            collection.
        name: Optional[str], default is None
            The name for the multiscale collection.
        **kwargs:
            Additional keyword arguments that will be passed to the `ArraySpec`
            constructor.
        """

        _paths = normalize_paths(arrays, paths)
        _chunks = normalize_chunks(arrays, chunks)
        attrs = CosemGroupMetadataV2.from_xarrays(arrays, _paths, name)

        array_specs = {
            key: CosemMultiscaleArray.from_xarray(arr, chunks=cnks, **kwargs)
            for arr, cnks, key in zip(arrays, _chunks, _paths)
        }

        return cls(attributes=attrs, members=array_specs)


from cellmap_schemas.multiscale.cosem import Group


def multiscale_group(
    *,
    arrays: dict[str, DataArray],
    chunks: Union[tuple[tuple[int, ...]], Literal["auto"]],
    **kwargs,
) -> Group:
    """
    Create a model of a COSEM-style multiscale group from a collection of
    DataArrays

    Parameters
    ----------

    arrays: dict[str, DataArray]
        The data to model.
    chunks: The chunks for each Zarr array in the group.

    """
    return Group.from_arrays(
        arrays=tuple(arrays.values()),
        paths=tuple(arrays.keys()),
        transforms=tuple(stt_from_array(a) for a in arrays),
        chunks=chunks,
        **kwargs,
    )


from xarray_ome_ngff.array_wrap import ArrayWrapperSpec, parse_wrapper


def create_datarray_array_wrapper(
    array: zarr.Array,
    *,
    array_wrapper: ZarrArrayWrapper
    | DaskArrayWrapper
    | ArrayWrapperSpec = DaskArrayWrapper(chunks="auto"),
) -> DataArray:
    wrapper_parsed = parse_wrapper(array_wrapper)
    if isinstance(wrapper_parsed, DaskArrayWrapper):
        use_dask = True
        chunks = wrapper_parsed.chunks
    else:
        use_dask = False
        chunks = "auto"
    return create_dataarray(array=array, chunks=chunks, use_dask=use_dask)


def create_dataarray(
    *,
    array: zarr.Array,
    use_dask: bool = True,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
) -> DataArray:
    """
    create a DataArray from an N5 container with cosem / cellmap metadata
    """
    if use_dask:
        array_wrapped = da.from_array(array, chunks=chunks)
    else:
        if chunks != "auto":
            msg = f"If use_dask is False, then chunks must be 'auto'. Got {chunks} instead."
            raise ValueError(msg)
        array_wrapped = array
    array_attrs = array.attrs.asdict()
    transform = STTransform(**array_attrs["transform"])
    coords = stt_to_coords(transform, array.shape)
    return DataArray(
        array_wrapped,
        coords=coords,
        dims=transform.axes,
        attrs=array.attrs.asdict(),
    )
