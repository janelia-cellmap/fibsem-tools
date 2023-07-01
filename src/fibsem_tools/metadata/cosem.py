from typing import Iterable, Literal, Optional, Sequence, Union

from pydantic import BaseModel
from xarray import DataArray
from pydantic_zarr import GroupSpec, ArraySpec
from fibsem_tools.metadata.transform import STTransform


class ScaleMetaV1(BaseModel):
    path: str
    transform: STTransform


class MultiscaleMetaV1(BaseModel):
    name: Optional[str]
    datasets: list[ScaleMetaV1]


class MultiscaleMetaV2(BaseModel):
    name: Optional[str]
    datasets: list[str]


class COSEMGroupMetadataV1(BaseModel):
    """
    Multiscale metadata used by COSEM for multiscale datasets saved in N5/Zarr groups.
    """

    multiscales: list[MultiscaleMetaV1]

    @classmethod
    def from_xarrays(
        cls,
        arrays: Sequence[DataArray],
        paths: Union[Sequence[str], Literal["auto"]],
        name: Optional[str] = None,
    ):
        """
        Generate multiscale metadata from a list or tuple of DataArrays.

        Parameters
        ----------

        arrays : list or tuple of xarray.DataArray
            The collection of arrays from which to generate multiscale metadata. These
            arrays are assumed to share the same `dims` attributes, albeit with varying
            `coords`.

        paths : Sequence of str or the string literal 'auto', default='auto'
            The name on the storage backend for each of the arrays in the multiscale
            collection. If 'auto', then names will be automatically generated using the
            format s0, s1, s2, etc

        name : str, optional
            The name for the multiresolution collection


        Returns an instance of COSEMGroupMetadataV1
        -------

        COSEMGroupMetadata
        """

        if paths == "auto":
            paths = [f"s{idx}" for idx in range(len(arrays))]

        multiscales = [
            MultiscaleMetaV1(
                name=name,
                datasets=[
                    ScaleMetaV1(path=path, transform=STTransform.from_xarray(array=arr))
                    for path, arr in zip(paths, arrays)
                ],
            )
        ]
        return cls(name=name, multiscales=multiscales, paths=paths)


class COSEMGroupMetadataV2(BaseModel):
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
        Generate multiscale metadata from a list or tuple of DataArrays.

        Parameters
        ----------

        arrays : list or tuple of xarray.DataArray
            The collection of arrays from which to generate multiscale metadata. These
            arrays are assumed to share the same `dims` attributes, albeit with varying
            `coords`.

        paths : list or tuple of str
            The name on the storage backend for each of the arrays in the multiscale
            collection.

        name : str, optional
            The name for the multiresolution collection

        Returns an instance of COSEMGroupMetadataV2
        -------

        COSEMGroupMetadata
        """
        if paths == "auto":
            paths = [f"s{idx}" for idx in enumerate(arrays)]

        multiscales = [
            MultiscaleMetaV2(
                name=name,
                datasets=paths,
            )
        ]
        return cls(name=name, multiscales=multiscales, paths=paths)


class CosemArrayAttrs(BaseModel):
    transform: STTransform


class CosemMultiscaleArray(ArraySpec):
    attrs: CosemArrayAttrs

    @classmethod
    def from_xarray(cls, array: DataArray, **kwargs):
        attrs = CosemArrayAttrs(transform=STTransform.from_xarray(array))
        return super().from_array(array, attrs=attrs, **kwargs)


class CosemMultiscaleGroupV1(GroupSpec):
    attrs: COSEMGroupMetadataV1
    items: dict[str, CosemMultiscaleArray]

    @classmethod
    def from_xarrays(
        cls,
        arrays: Iterable[DataArray],
        paths: Union[Sequence[str], Literal["auto"]] = "auto",
        name: Optional[str] = None,
        **kwargs,
    ):

        if paths == "auto":
            paths = [f"s{idx}" for idx in range(len(arrays))]

        attrs = COSEMGroupMetadataV1.from_xarrays(arrays, paths, name)

        array_specs = {
            k: CosemMultiscaleArray.from_xarray(arr, **kwargs)
            for k, arr in zip(paths, arrays)
        }

        return cls(attrs=attrs, items=array_specs)


class CosemMultiscaleGroupV2(GroupSpec):
    attrs: COSEMGroupMetadataV2
    items: dict[str, ArraySpec[CosemArrayAttrs]]

    @classmethod
    def from_xarrays(
        cls,
        arrays: Iterable[DataArray],
        paths: Union[Sequence[str], Literal["auto"]] = "auto",
        name: Optional[str] = None,
        **kwargs,
    ):

        if paths == "auto":
            paths = [f"s{idx}" for idx in range(len(arrays))]

        attrs = COSEMGroupMetadataV2.from_xarrays(arrays, paths, name)

        array_specs = {
            k: CosemMultiscaleArray.from_xarray(arr, **kwargs)
            for k, arr in zip(paths, arrays)
        }

        return cls(attrs=attrs, items=array_specs)
