from typing import Iterable, List, Sequence

import numpy as np
from pydantic import BaseModel, PositiveInt, ValidationError, validator
from xarray import DataArray
from pydantic_zarr.core import GroupSpec, ArraySpec
from fibsem_tools.metadata.transform import STTransform


class PixelResolution(BaseModel):
    """
    PixelResolution attribute used by the Saalfeld lab. The dimensions attribute
    contains a list of scales that define the grid spacing of the data, in F-order.
    """

    dimensions: Sequence[float]
    unit: str


# todo: validate argument lengths
class NeuroglancerN5GroupMetadata(BaseModel):
    """
    Metadata to enable displaying an N5 group containing several datasets
    as a multiresolution dataset in neuroglancer.
    see https://github.com/google/neuroglancer/issues/176#issuecomment-553027775
    Axis properties will be indexed in the opposite order of C-contiguous axis indexing.
    """

    axes: Sequence[str]
    units: Sequence[str]
    scales: Sequence[Sequence[PositiveInt]]
    pixelResolution: PixelResolution

    @classmethod
    def from_xarrays(cls, arrays: Sequence[DataArray]) -> "NeuroglancerN5GroupMetadata":
        """
        Create neuroglancer-compatible N5 metadata from a collection of DataArrays.

        Parameters
        ----------

        arrays : list or tuple of xarray.DataArray
            The collection of arrays from which to generate multiscale metadata. These
            arrays are assumed to share the same `dims` attributes, albeit with varying
            `coords`.

        Returns
        -------

        NeuroglancerN5GroupMetadata
        """
        transforms = [
            STTransform.from_xarray(array, reverse_axes=True) for array in arrays
        ]
        pixelresolution = PixelResolution(
            dimensions=transforms[0].scale, unit=transforms[0].units[0]
        )
        scales: List[List[int]] = [
            np.round(np.divide(t.scale, transforms[0].scale)).astype("int").tolist()
            for t in transforms
        ]
        return cls(
            axes=transforms[0].axes,
            units=transforms[0].units,
            scales=scales,
            pixelResolution=pixelresolution,
        )


class NeuroglancerN5Group(GroupSpec):
    attrs: NeuroglancerN5GroupMetadata

    @validator("members")
    def validate_members(cls, v: dict[str, ArraySpec]):
        # check that the names of the arrays are s0, s1, s2, etc
        for key, spec in v.items():
            assert key.startswith("s")
            try:
                int(key.split("s")[-1])
            except ValueError as valerr:
                raise ValidationError from valerr

        assert len(set(a.dtype for a in v.values())) == 1
        return v

    @classmethod
    def from_xarrays(
        cls, arrays: Iterable[DataArray], chunks: tuple[int, ...], **kwargs
    ) -> "NeuroglancerN5Group":
        array_specs = {
            f"s{idx}": ArraySpec.from_array(arr, chunks=chunks, **kwargs)
            for idx, arr in enumerate(arrays)
        }
        attrs = NeuroglancerN5GroupMetadata.from_xarrays(arrays)
        return cls(attrs=attrs, members=array_specs)
