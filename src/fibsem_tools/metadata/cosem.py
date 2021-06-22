from pydantic import BaseModel
from typing import Sequence, List, Optional
from xarray import DataArray
from .transform import SpatialTransform

class ScaleMeta(BaseModel):
    path: str
    transform: SpatialTransform


class MultiscaleMeta(BaseModel):
    datasets: Sequence[ScaleMeta]


class COSEMGroupMetadata(BaseModel):
    """
    Multiscale metadata used by COSEM for multiscale datasets saved in N5/Zarr groups.
    """

    name: str
    multiscales: Sequence[MultiscaleMeta]

    @classmethod
    def fromDataArrays(cls, dataarrays: Sequence[DataArray], name: str, paths: Optional[Sequence[str]]):
        """
        Generate multiscale metadata from a list or tuple of DataArrays.

        Parameters
        ----------

        dataarrays : list or tuple of xarray.DataArray
            The collection of arrays from which to generate multiscale metadata. These arrays are assumed to share the same `dims` attributes, albeit with varying `coords`.

        name : str
            The name for the multiresolution collection
        
        paths : list or tuple of str or None, default=None
            The name on the storage backend for each of the arrays in the multiscale collection. If None, the `name` attribute of each array in `dataarrays` will be used.

        Returns
        -------

        COSEMGroupMetadata
        """
        
        if paths is None:
            _paths: Sequence[str] = [str(d.name) for d in dataarrays]
        else:
            _paths = paths

        multiscales = [
            MultiscaleMeta(
                datasets=[
                    ScaleMeta(path=path, transform=SpatialTransform.fromDataArray(arr))
                    for path, arr in zip(paths, dataarrays)
                ]
            )
        ]
        return cls(name=name, multiscales=multiscales, paths=_paths)