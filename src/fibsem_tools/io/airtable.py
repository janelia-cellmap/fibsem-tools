import os
from typing import Dict, Literal, Optional, Tuple, Union
from pydantic import BaseModel
from pyairtable import Api
from fibsem_tools.metadata.transform import STTransform
from dotenv import load_dotenv

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
    title: str

    def to_stt(self):
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


def get_airbase():
    """
    Gets a pyairtable.Api.base object, given two environment variables: AIRTABLE_API_KEY and
    AIRTABLE_BASE_ID.

    Returns
    -------
    pyairtable.Api.base
    """
    return Api(os.environ["AIRTABLE_API_KEY"]).base(os.environ["AIRTABLE_BASE_ID"])


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


def class_encoding_from_airtable_by_image(image_name: str) -> Dict[str, int]:
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


def image_from_airtable(image_name: str) -> ImageRow:
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
    result = airbase.table("image").first(formula='{name} = "' + image_name + '"')
    if result is None:
        raise ValueError(f"Airtable does not contain a record named {image_name}")
    else:
        fields = result["fields"]
        try:
            return ImageRow(**fields, id=result["id"])
        except KeyError as e:
            raise ValueError(f"Missing field in airtable: {e}")


def coords_from_airtable(
    image_name: str, shape: Union[Literal["auto"], Tuple[int, ...]] = "auto"
) -> STTransform:
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
    img = image_from_airtable(image_name)
    if shape == "auto":
        _shape = (img.size_z_pix, img.size_y_pix, img.size_x_pix)
    else:
        _shape = shape
    return img.to_stt().to_coords(shape=_shape)
