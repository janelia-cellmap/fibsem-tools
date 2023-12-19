import os
from typing import Dict, Literal, Optional, Sequence, Tuple, Union
from pydantic import BaseModel
import pyairtable as pyat
from xarray import DataArray
from fibsem_tools.io.mrc import to_dask
from fibsem_tools.metadata.transform import STTransform
from dotenv import load_dotenv
import warnings
from fibsem_tools.types import ArrayLike
from fibsem_tools import read

load_dotenv()


class ImageRow(BaseModel):
    """
    A pydantic model that represents a row from the airtable "image" table.

    Attributes
    ----------
    id: str
        The unique identifier of this row.
    size_x_pix: int
        The length of the x axis of the image, in array indices.
    size_y_pix: int
        The length of the y axis of the image, in array indices.
    size_z_pix: int
        The length of the z axis of the image, in array indices.
    resolution_x_nm: float
        The spacing between samples along the x axis of the image, in nanometers.
    resolution_y_nm: float
        The spacing between samples along the y axis of the image, in nanometers.
    resolution_z_nm: float
        The spacing between samples along the z axis of the image, in nanometers.
    offset_x_nm: float
        The position of the first sample of the x axis of the image, in nanometers.
    offset_y_nm: float
        The position of the first sample of the y axis of the image, in nanometers.
    offset_z_nm: float
        The position of the first sample of the z axis of the image, in nanometers.
    value_type: Literal['scalar', 'label']
        The type of the values in the image. For label images, like crops and
        segmentations, this should be `label`. For intensity-based images, like FIB-SEM
        data, this value should be `scalar`.
    image_type: str
        The category of image, e.g. em data, human segmentation, etc.
    title: str
        The name of the image.
    location: str
        The location of this image in storage
    format: Literal['zarr', 'n5', 'mrc', 'hdf5', 'tiff']
        The format this image is stored in.
    """

    id: str
    size_x_pix: int
    size_y_pix: int
    size_z_pix: int
    resolution_x_nm: float
    resolution_y_nm: float
    resolution_z_nm: float
    offset_x_nm: float
    offset_y_nm: float
    offset_z_nm: float
    value_type: Literal["scalar", "label"]
    image_type: str
    title: Optional[str]
    location: str
    format: Literal['zarr', 'n5', 'mrc', 'hdf5', 'tiff']

    def to_stt(self) -> STTransform:
        """
        Convert this image to a STTransform.

        Returns
        -------
        STTransform
        """
        return STTransform(
            order="C",
            units=("nm", "nm", "nm"),
            axes=("z", "y", "x"),
            scale=(self.resolution_z_nm, self.resolution_y_nm, self.resolution_x_nm),
            translate=(self.offset_z_nm, self.offset_y_nm, self.offset_x_nm),
        )


def get_airbase() -> pyat.Base:
    """
    Gets a pyairtable.Api.base object, given two environment variables: AIRTABLE_API_KEY and
    AIRTABLE_BASE_ID.

    Returns
    -------
    pyairtable.Api.base
    """
    return pyat.Api(os.environ["AIRTABLE_API_KEY"]).base(os.environ["AIRTABLE_BASE_ID"])


class AnnotationState(BaseModel):
    """
    The state of an annotation for a particular class. Corresponds to rows in the
    "annotation_type" on airtable.

    Attributes
    ----------
    present: Optional[bool]
        Whether this class is present or absent in a crop.
    annotated: Optional[bool]
        Whether this class is annotated in a crop
    sparse: Optional[bool]
        Whether this class is sparse or not (i.e., dense) in a crop.
    """

    present: Optional[bool]
    annotated: Optional[bool]
    sparse: Optional[bool]


def class_encoding_by_image(image_name: str) -> Dict[str, int]:
    """
    Get a class encoding from airtable. A class encoding is a dict where the keys are
    strings (more specifically, class ids), and the values are integers
    (more specifically, the canonical value used when annotating that class).

    Parameters
    ----------
    image_name: str
        The name of the annotated image.

    Returns
    -------
    Dict[str, int]
        A mapping from class ids to integers.

    """
    airbase = get_airbase()
    # airtable does not return False for an unchecked checkbox, so merging the result of
    # an airtable query with this dict will map empty fields to False.
    annotation_defaults = {"present": False, "annotated": False, "sparse": False}
    annotation_table = airbase.table("annotation")
    annotation_types = airbase.table("annotation_type").all(
        fields=["name", "present", "annotated", "sparse"]
    )
    annotation_types_parsed = {
        a["id"]: AnnotationState(**{**annotation_defaults, **a["fields"]})
        for a in annotation_types
    }
    result = annotation_table.first(formula='{image} = "' + image_name + '"')
    if result is None:
        raise ValueError(f"Airtable does not contain a record named {image_name}")
    else:
        fields = result["fields"]
        out = {}
        for key in tuple(fields.keys()):
            try:
                value_str, *rest = key.split("_")
                value = int(value_str)
                # raises if the prefix is not an int
                annotation_type = annotation_types_parsed[fields[key][0]]
                if annotation_type.annotated:
                    out["_".join(rest)] = int(value)
            except ValueError:
                # the field name was not of the form <number>_<class>
                pass
    return out


def query_image_by_name(image_name: str) -> ImageRow:
    """
    Get metadata about an image from airtable.

    Parameters
    ----------
    image_name: str
        The name of the image.

    Returns
    -------
    ImageRow
    """
    airbase = get_airbase()
    result = airbase.table("image").all(formula='{name} = "' + image_name + '"')
    if len(result) > 1:
        raise ValueError(
            f"Retrieved {len(result)} images with the name {image_name}. Ensure that the name is unique."
        )
    if result is None:
        raise ValueError(f"Airtable does not contain a record named {image_name}")
    else:
        fields = result[0]["fields"]
        try:
            return ImageRow(**fields, id=result[0]["id"])
        except KeyError as e:
            raise ValueError(f"Missing field in airtable: {e}")


def coords_from_airtable(
    image_name: str, shape: Union[Literal["auto"], Tuple[int, ...]] = "auto"
) -> Sequence[DataArray]:
    """
    Get xarray-ready coordinates for an array based on an entry in airtable.

    Parameters
    ----------
    image_name: str
        The name of the image in airtable.
    shape: Literal['auto'] | Tuple[int,...]
        The shape of the array. If this is 'auto' (the default), then the size of the
        image will be inferred from airtable.

    Returns
    -------
    List[DataArray]
    """
    img = query_image_by_name(image_name)
    _shape: Tuple[int, ...]

    if shape == "auto":
        _shape = (img.size_z_pix, img.size_y_pix, img.size_x_pix)
    else:
        _shape = shape

    return img.to_stt().to_coords(shape=_shape)


def to_xarray(
        image: ImageRow, 
        coords: Union[Literal['from_airtable', 'from_file'], Sequence[DataArray]] = 'from_airtable') -> DataArray:
    from urllib.parse import urlparse, unquote
    # only local for now
    loc_parsed = urlparse(image.location)
    if loc_parsed.scheme != 'file':
        raise ValueError(f'Cannot deal with non-file urls. Got {loc_parsed.scheme}')
    path = unquote(loc_parsed.path)

    item = read(path)
    if not isinstance(item, ArrayLike):
        raise ValueError(
            f"Reading {image.location} produced an instance of {type(item)}, which is not an array."
        )
    
    airtable_coords = image.to_stt().to_coords(item.shape)
    if image.format == 'mrc':
        from fibsem_tools.io.mrc import to_xarray, infer_coords
        file_coords = infer_coords(item)
        stt_file = STTransform.from_coords(file_coords)
        stt_airtable = STTransform.from_coords(airtable_coords)
        if stt_file != stt_airtable:
            warnings.warn(
                f"The scale + translation representation of the coordinates from airtable ({stt_file}) "
                f"does not match the scale + translation representation inferred from the image data ({stt_airtable})"
                )
        if coords == 'from_airtable':
            return to_xarray(item, coords=airtable_coords)
        elif coords == 'from_file':
            return to_xarray(item, coords=file_coords)
        else:
            return to_xarray(item, coords=coords)
    else:
        raise NotImplementedError    