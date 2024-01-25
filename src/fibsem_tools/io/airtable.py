from __future__ import annotations
import os
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Union
import datatree
from pydantic import BaseModel
import pyairtable as pyat
from pyairtable.api.types import RecordDict
from xarray import DataArray
import xarray
from fibsem_tools.metadata.transform import STTransform
from dotenv import load_dotenv
import warnings
from fibsem_tools.types import Arrayish
from fibsem_tools import read
from typing_extensions import Self
from pydantic import Field
from urllib.parse import urlparse, unquote
import fibsem_tools.io.mrc as mrc

load_dotenv()
ValueType = Literal["label", "scalar"]
ImageType = Literal[
    "em", "human_segmentation", "lm", "ml_prediction", "ml_segmentation"
]
ImageFormat = Literal["mrc", "n5", "zarr", "tif", "hdf5"]


def get_airbase() -> pyat.Base:
    """
    Gets a pyairtable.Api.base object, given two environment variables: AIRTABLE_API_KEY and
    AIRTABLE_BASE_ID.

    Returns
    -------
    pyairtable.Api.base
    """
    return pyat.Api(os.environ["AIRTABLE_API_KEY"]).base(os.environ["AIRTABLE_BASE_ID"])


class ImageInsert(BaseModel):
    """
    A pydantic model that represents a row that can be inserted into the airtable "image" table.
    """

    name: Optional[str]
    collection: Optional[List[str]]
    title: Optional[str]
    location: Optional[str]
    image_type: Optional[ImageType]
    value_type: Optional[ValueType]
    source_image: Optional[List[str]]
    reconstruction_protocol: Optional[List[str]]
    size_x_pix: Optional[int]
    size_y_pix: Optional[int]
    size_z_pix: Optional[int]
    resolution_x_nm: Optional[float]
    resolution_y_nm: Optional[float]
    resolution_z_nm: Optional[float]
    offset_x_nm: Optional[float]
    offset_y_nm: Optional[float]
    offset_z_nm: Optional[float]
    fibsem_imaging: Optional[List[str]]
    light_imaging: Optional[List[str]]
    xray_imaging: Optional[List[str]]
    tem_imaging: Optional[List[str]]
    annotations: Optional[List[str]]
    clas: Optional[List[str]] = Field(alias="class")
    notes: Optional[str]
    format: Optional[ImageFormat]

    class Config:
        allow_population_by_field_name = True

    def to_stt(self: ImageInsert, order: Literal["C", "F"] = "C"):
        scale = [self.resolution_x_nm, self.resolution_y_nm, self.resolution_z_nm]
        translate = [self.offset_x_nm, self.offset_y_nm, self.offset_z_nm]
        units = ["nm", "nm", "nm"]
        axes = ["x", "y", "z"]

        if order == "C":
            orderer = slice(None, None, -1)
        else:
            orderer = slice(0, None, 1)
        return STTransform(
            order=order,
            scale=scale[orderer],
            translate=translate[orderer],
            units=units,
            axes=axes[orderer],
        )

    @classmethod
    def from_xarray(
        cls,
        data: xarray.DataArray,
        *,
        name: Optional[str] = None,
        collection: Optional[str] = None,
        location: Optional[str] = None,
        value_type: Optional[ValueType] = None,
        image_type: Optional[ImageType] = None,
        source_image: Optional[List[str]] = None,
        reconstruction_protocol: Optional[List[str]] = None,
        fibsem_imaging: Optional[List[str]] = None,
        light_imaging: Optional[List[str]] = None,
        xray_imaging: Optional[List[str]] = None,
        tem_imaging: Optional[List[str]] = None,
        annotations: Optional[List[str]] = None,
        clas: Optional[List[str]] = None,
        notes: Optional[str] = None,
        format: Optional[str] = None,
        title: Optional[str] = None,
    ):
        if set(data.dims) != {"x", "y", "z"}:
            msg = f'{cls.__name__} is only compatible with 3D data where the dimensions are named "x", "y", and "z". Got dimensions {data.dims} instead.'
            raise ValueError(msg)

        axis_indices = {key: data.dims.index(key) for key in ("x", "y", "z")}

        size_x_pix = data.shape[axis_indices["x"]]
        size_y_pix = data.shape[axis_indices["y"]]
        size_z_pix = data.shape[axis_indices["z"]]

        tx = STTransform.from_xarray(data)
        # todo: add convenience function that returns a new STTransform with different units and correspondingly adjusted
        # translate and scale parameters, to convert from e.g. microns to nm
        # todo: normalize this with pint
        if set(tx.units) not in ({"nm"}, {"nanometer"}, {"nanometers"}):
            msg = f"All of the units of the data must be nm. Got {tx.units} instead."
            raise ValueError(msg)

        resolution_x_nm = tx.scale[axis_indices["x"]]
        resolution_y_nm = tx.scale[axis_indices["y"]]
        resolution_z_nm = tx.scale[axis_indices["z"]]

        offset_x_nm = tx.translate[axis_indices["x"]]
        offset_y_nm = tx.translate[axis_indices["y"]]
        offset_z_nm = tx.translate[axis_indices["z"]]

        return cls(
            name=name,
            collection=collection,
            size_x_pix=size_x_pix,
            size_y_pix=size_y_pix,
            size_z_pix=size_z_pix,
            offset_x_nm=offset_x_nm,
            offset_y_nm=offset_y_nm,
            offset_z_nm=offset_z_nm,
            resolution_x_nm=resolution_x_nm,
            resolution_y_nm=resolution_y_nm,
            resolution_z_nm=resolution_z_nm,
            location=location,
            image_type=image_type,
            value_type=value_type,
            source_image=source_image,
            reconstruction_protocol=reconstruction_protocol,
            fibsem_imaging=fibsem_imaging,
            light_imaging=light_imaging,
            xray_imaging=xray_imaging,
            tem_imaging=tem_imaging,
            annotations=annotations,
            clas=clas,
            notes=notes,
            title=title,
            format=format,
        )

    def to_xarray(
        self,
        coords: Union[
            Literal["from_airtable", "from_file"], Sequence[DataArray]
        ] = "from_airtable",
    ) -> Union[xarray.DataArray, datatree.DataTree]:
        # only local for now
        loc_parsed = urlparse(self.location)
        if loc_parsed.scheme != "file":
            msg = f"Cannot deal with non-file urls. Got {loc_parsed.scheme}"
            raise ValueError(msg)
        path = unquote(loc_parsed.path)

        item = read(path)
        if not isinstance(item, Arrayish):
            msg = f"Reading {self.location} produced an instance of {type(item)}, which is not an array."
            raise ValueError(msg)

        airtable_coords = self.to_stt().to_coords(item.shape)
        if self.format == "mrc":
            file_coords = mrc.infer_coords(item)
            stt_file = STTransform.from_coords(file_coords)
            stt_airtable = STTransform.from_coords(airtable_coords)
            if stt_file != stt_airtable:
                msg = (
                    f"The scale + translation representation of the coordinates from airtable ({stt_file}) "
                    f"does not match the scale + translation representation inferred from the image data ({stt_airtable})"
                )
                warnings.warn(msg)
            if coords == "from_airtable":
                return mrc.to_xarray(item, coords=airtable_coords)
            elif coords == "from_file":
                return mrc.to_xarray(item, coords=file_coords)
            else:
                return mrc.to_xarray(item, coords=coords)
        else:
            raise NotImplementedError


class ImageSelect(ImageInsert):
    """
    A pydantic model that represents a row selected the airtable "image" table.

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
    location: Optional[str]
        The location of this image in storage
    format: Optional[Literal['zarr', 'n5', 'mrc', 'hdf5', 'tiff']]
        The format this image is stored in.
    """

    # todo: fix shadowing of builtins.id
    id: str
    createdTime: str

    @classmethod
    def from_airtable(cls, payload: RecordDict) -> Self:
        fields = payload["fields"]
        if "class" in fields:
            clas = fields.pop("class")
            fields["clas"] = clas
        return cls(id=payload["id"], createdTime=payload["createdTime"], **fields)


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

    present: Optional[bool] = False
    annotated: Optional[bool] = False
    sparse: Optional[bool] = False


class LabelledAnnotationState(AnnotationState):
    numeric_label: int


class AnnotationSelect(BaseModel):
    id: str
    createdTime: str
    name: str
    images: Union[list[str], list[ImageSelect]]
    ecs: Optional[AnnotationState] = Field(alias="1_ecs")
    pm: Optional[AnnotationState] = Field(alias="2_pm")
    mito_mem: Optional[AnnotationState] = Field(alias="3_mito_mem")
    mito_lum: Optional[AnnotationState] = Field(alias="4_mito_lim")
    mito_ribo: Optional[AnnotationState] = Field(alias="5_mito_ribo")
    golgi_mem: Optional[AnnotationState] = Field(alias="6_golgi_mem")
    golgi_lum: Optional[AnnotationState] = Field(alias="7_golgi_lum")
    ves_mem: Optional[AnnotationState] = Field(alias="8_ves_mem")
    ves_lum: Optional[AnnotationState] = Field(alias="9_ves_mem")
    endo_mem: Optional[AnnotationState] = Field(alias="10_endo_mem")
    endo_lum: Optional[AnnotationState] = Field(alias="11_endo_lum")
    lyso_mem: Optional[AnnotationState] = Field(alias="12_lyso_mem")
    lyso_lum: Optional[AnnotationState] = Field(alias="13_lyso_lum")
    ld_mem: Optional[AnnotationState] = Field(alias="14_ld_mem")
    ld_lum: Optional[AnnotationState] = Field(alias="15_ld_lum")
    er_mem: Optional[AnnotationState] = Field(alias="16_er_mem")
    er_lum: Optional[AnnotationState] = Field(alias="17_er_lum")
    eres_mem: Optional[AnnotationState] = Field(alias="18_eres_mem")
    eres_lum: Optional[AnnotationState] = Field(alias="19_eres_lum")
    ne_mem: Optional[AnnotationState] = Field(alias="20_ne_mem")
    ne_lum: Optional[AnnotationState] = Field(alias="21_ne_lum")
    np_out: Optional[AnnotationState] = Field(alias="22_np_out")
    np_in: Optional[AnnotationState] = Field(alias="23_np_in")
    hchrom: Optional[AnnotationState] = Field(alias="24_hchrom")
    nhchrom: Optional[AnnotationState] = Field(alias="25_nhchrom")
    echrom: Optional[AnnotationState] = Field(alias="26_echrom")
    nechrom: Optional[AnnotationState] = Field(alias="27_nechrom")
    nucpl: Optional[AnnotationState] = Field(alias="28_nucpl")
    nucleo: Optional[AnnotationState] = Field(alias="29_nucleo")
    mt_out: Optional[AnnotationState] = Field(alias="30_mt_out")
    cent: Optional[AnnotationState] = Field(alias="31_cent")
    cent_dapp: Optional[AnnotationState] = Field(alias="32_cent_dapp")
    cent_sdapp: Optional[AnnotationState] = Field(alias="33_cent_sdapp")
    ribo: Optional[AnnotationState] = Field(alias="34_ribo")
    cyto: Optional[AnnotationState] = Field(alias="35_cyto")
    mt_in: Optional[AnnotationState] = Field(alias="36_mt_in")
    nuc: Optional[AnnotationState] = Field(alias="37_nuc")
    vim: Optional[AnnotationState] = Field(alias="38_vim")
    glyco: Optional[AnnotationState] = Field(alias="39_glyco")
    golgi: Optional[AnnotationState] = Field(alias="40_golgi")
    ves: Optional[AnnotationState] = Field(alias="41_ves")
    endo: Optional[AnnotationState] = Field(alias="42_endo")
    lyso: Optional[AnnotationState] = Field(alias="43_lyso")
    ld: Optional[AnnotationState] = Field(alias="44_ld")
    rbc: Optional[AnnotationState] = Field(alias="45_rbc")
    eres: Optional[AnnotationState] = Field(alias="46_eres")
    perox_mem: Optional[AnnotationState] = Field(alias="47_perox_mem")
    perox_lum: Optional[AnnotationState] = Field(alias="48_perox_lum")
    perox: Optional[AnnotationState] = Field(alias="49_perox")
    mito: Optional[AnnotationState] = Field(alias="50_mito")
    er: Optional[AnnotationState] = Field(alias="51_er")
    ne: Optional[AnnotationState] = Field(alias="52_ne")
    np: Optional[AnnotationState] = Field(alias="53_np")
    chrom: Optional[AnnotationState] = Field(alias="54_chrom")
    mt: Optional[AnnotationState] = Field(alias="55_mt")
    isg_mem: Optional[AnnotationState] = Field(alias="56_ism_mem")
    isg_lum: Optional[AnnotationState] = Field(alias="57_isg_lum")
    isg_ins: Optional[AnnotationState] = Field(alias="58_isg_ins")
    isg: Optional[AnnotationState] = Field(alias="59_isg")
    cell: Optional[AnnotationState] = Field(alias="60_cell")
    actin: Optional[AnnotationState] = Field(alias="61_actin")
    tbar: Optional[AnnotationState] = Field(alias="62_tbar")
    bm: Optional[AnnotationState] = Field(alias="63_bm")
    er_mem_all: Optional[AnnotationState] = Field(alias="64_er_mem_all")
    ne_mem_all: Optional[AnnotationState] = Field(alias="65_ne_mem_all")
    cent_all: Optional[AnnotationState] = Field(alias="66_cent_all")
    chloroplast_mem: Optional[AnnotationState] = Field(alias="67_chloroplast_mem")
    chloroplast_lum: Optional[AnnotationState] = Field(alias="68_chloroplast_lum")
    chloroplast_sg: Optional[AnnotationState] = Field(alias="69_chloroplast_sg")
    chloroplast: Optional[AnnotationState] = Field(alias="70_chloroplast")
    vacuole_mem: Optional[AnnotationState] = Field(alias="71_vacuole_mem")
    vacuole_lum: Optional[AnnotationState] = Field(alias="72_vacuole_lum")
    vacuole: Optional[AnnotationState] = Field(alias="73_vacuole")
    plasmodesmata: Optional[AnnotationState] = Field(alias="74_plasmodesmata")

    def annotation_states(self) -> Dict[str, AnnotationState]:
        """
        Return the annotation states as a dict.

        Returns
        -------
        Dict[str, int]
            A mapping from class name to AnnotationState.

        """
        result = {}
        for key, value in filter(
            lambda v: isinstance(v[1], AnnotationState), self.__dict__.items()
        ):
            result[key] = value

        return result

    @classmethod
    def from_airtable(cls, data: RecordDict, *, resolve_images: bool = False):
        if resolve_images:
            data["fields"]["images"] = [
                select_image_by_id(im_id) for im_id in data["fields"]["images"]
            ]

        return cls(id=data["id"], createdTime=data["createdTime"], **data["fields"])


def select_annotation_by_name(name: str, resolve_images=False) -> AnnotationSelect:
    airbase = get_airbase()
    annotations = airbase.table("annotation")
    class_query_result = airbase.table("class").all(fields=["field_name"])
    class_names = [c["fields"]["field_name"] for c in class_query_result]
    annotation_types = airbase.table("annotation_type").all(
        fields=["name", "present", "annotated", "sparse"]
    )
    annotation_types_parsed = {
        a["id"]: AnnotationState(**a["fields"]) for a in annotation_types
    }
    formula = " ".join(["{name}", "=", f'"{name}"'])
    annotation_query_result = annotations.first(formula=formula)
    if annotation_query_result is None:
        msg = f"Airtable does not contain a record in the annotation table identified by {name}"
        raise ValueError(msg)
    else:
        fields = annotation_query_result["fields"]
        out: RecordDict = {}
        out["createdTime"] = annotation_query_result["createdTime"]
        out["id"] = annotation_query_result["id"]
        out["fields"] = {}
        for key, field in fields.items():
            value_str, *rest = key.split("_")
            if "_".join(rest) in class_names:
                # raises if the prefix is not an int
                value = int(value_str)
                annotation_type = annotation_types_parsed[field[0]].dict()
                out["fields"][key] = LabelledAnnotationState(
                    **annotation_type, numeric_label=value
                )
            else:
                out["fields"][key] = field
    return AnnotationSelect.from_airtable(out, resolve_images=resolve_images)


def select_image_by_id(id: str):
    airbase = get_airbase()
    result = airbase.table("image").get(id)
    if result is None:
        msg = f"Airtable does not contain a record with id={id}"
        raise ValueError(msg)
    return ImageSelect.from_airtable(result)


def select_image_by_collection_and_name(identifier: str) -> ImageSelect:
    """
    Get metadata about an images from airtable.

    Parameters
    ----------
    image_name: str
        The name of the image.

    collection: str
        The name of the collection which contains the image.

    Returns
    -------
    ImageRow
    """
    airbase = get_airbase()
    formula = " ".join(["{collection/name}", "=", f'"{identifier}"'])
    result = airbase.table("image").first(formula=formula)
    if result is None:
        msg = f"Airtable does not contain a record identified by {identifier}"
        raise ValueError(msg)
    return ImageSelect.from_airtable(result)


def insert_images(images: Iterable[ImageInsert]):
    """
    Insert new images to Airtable
    """

    base = get_airbase()
    return base.table("image").batch_create([im.dict(by_alias=True) for im in images])


def upsert_images_by_location(images: Iterable[ImageInsert]):
    """
    Insert images to Airtable. For each image, if another image exists with the same location,
    that record will be updated instead.
    """

    base = get_airbase()
    return base.table("image").batch_upsert(
        [{"fields": im.dict(by_alias=True)} for im in images], key_fields=["location"]
    )
