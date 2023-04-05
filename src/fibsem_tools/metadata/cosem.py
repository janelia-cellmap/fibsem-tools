from typing import Optional, Sequence

from pydantic import BaseModel
from xarray import DataArray

from fibsem_tools.metadata.transform import STTransform


class ScaleMetaV1(BaseModel):
    path: str
    transform: STTransform


class MultiscaleMetaV1(BaseModel):
    name: Optional[str]
    datasets: Sequence[ScaleMetaV1]


class MultiscaleMetaV2(BaseModel):
    name: Optional[str]
    datasets: Sequence[str]


class COSEMGroupMetadataV1(BaseModel):
    """
    Multiscale metadata used by COSEM for multiscale datasets saved in N5/Zarr groups.
    """

    multiscales: Sequence[MultiscaleMetaV1]

    @classmethod
    def fromDataArrays(
        cls,
        arrays: Sequence[DataArray],
        paths: Sequence[str],
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

        paths : list or tuple of str or None, default=None
            The name on the storage backend for each of the arrays in the multiscale
            collection.

        name : str, optional
            The name for the multiresolution collection


        Returns an instance of COSEMGroupMetadataV1
        -------

        COSEMGroupMetadata
        """

        multiscales = [
            MultiscaleMetaV1(
                name=name,
                datasets=[
                    ScaleMetaV1(
                        path=path, transform=STTransform.fromDataArray(array=arr)
                    )
                    for path, arr in zip(paths, arrays)
                ],
            )
        ]
        return cls(name=name, multiscales=multiscales, paths=paths)


class COSEMGroupMetadataV2(BaseModel):
    """
    Multiscale metadata used by COSEM for multiscale datasets saved in N5/Zarr groups.
    """

    multiscales: Sequence[MultiscaleMetaV2]

    @classmethod
    def fromDataArrays(
        cls,
        arrays: Sequence[DataArray],
        paths: Sequence[str],
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

        multiscales = [
            MultiscaleMetaV2(
                name=name,
                datasets=paths,
            )
        ]
        return cls(name=name, multiscales=multiscales, paths=paths)
