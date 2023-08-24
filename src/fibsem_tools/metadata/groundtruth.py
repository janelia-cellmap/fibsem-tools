from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Dict, Generic, List, Literal, Optional, TypeVar, Union
from pydantic_zarr import GroupSpec, ArraySpec
from pydantic import BaseModel, root_validator
from pydantic.generics import GenericModel


class StrictBase(BaseModel):
    class Config:
        extra = "forbid"


T = TypeVar("T")


class CellmapWrapper(StrictBase, GenericModel, Generic[T]):
    cellmap: T


class AnnotationWrapper(StrictBase, GenericModel, Generic[T]):
    annotation: T


class InstanceName(StrictBase):
    long: str
    short: str


class Annotated(str, Enum):
    dense: str = "dense"
    sparse: str = "sparse"
    empty: str = "empty"


class AnnotationState(StrictBase):
    present: bool
    annotated: Annotated


class Label(StrictBase):
    value: int
    name: InstanceName
    annotationState: AnnotationState
    count: Optional[int]


class LabelList(StrictBase):
    labels: List[Label]
    annotation_type: AnnotationType = "semantic"


classNameDict = {
    1: InstanceName(short="ECS", long="Extracellular Space"),
    2: InstanceName(short="Plasma membrane", long="Plasma membrane"),
    3: InstanceName(short="Mito membrane", long="Mitochondrial membrane"),
    4: InstanceName(short="Mito lumen", long="Mitochondrial lumen"),
    5: InstanceName(short="Mito DNA", long="Mitochondrial DNA"),
    6: InstanceName(short="Golgi Membrane", long="Golgi apparatus membrane"),
    7: InstanceName(short="Golgi lumen", long="Golgi apparatus lumen"),
    8: InstanceName(short="Vesicle membrane", long="Vesicle membrane"),
    9: InstanceName(short="Vesicle lumen", long="VesicleLumen"),
    10: InstanceName(short="MVB membrane", long="Multivesicular body membrane"),
    11: InstanceName(short="MVB lumen", long="Multivesicular body lumen"),
    12: InstanceName(short="Lysosome membrane", long="Lysosome membrane"),
    13: InstanceName(short="Lysosome lumen", long="Lysosome membrane"),
    14: InstanceName(short="LD membrane", long="Lipid droplet membrane"),
    15: InstanceName(short="LD lumen", long="Lipid droplet lumen"),
    16: InstanceName(short="ER membrane", long="Endoplasmic reticulum membrane"),
    17: InstanceName(short="ER lumen", long="Endoplasmic reticulum membrane"),
    18: InstanceName(
        short="ERES membrane", long="Endoplasmic reticulum exit site membrane"
    ),
    19: InstanceName(short="ERES lumen", long="Endoplasmic reticulum exit site lumen"),
    20: InstanceName(short="NE membrane", long="Nuclear envelope membrane"),
    21: InstanceName(short="NE lumen", long="Nuclear envelope lumen"),
    22: InstanceName(short="Nuclear pore out", long="Nuclear pore out"),
    23: InstanceName(short="Nuclear pore in", long="Nuclear pore in"),
    24: InstanceName(short="HChrom", long="Heterochromatin"),
    25: InstanceName(short="NHChrom", long="Nuclear heterochromatin"),
    26: InstanceName(short="EChrom", long="Euchromatin"),
    27: InstanceName(short="NEChrom", long="Nuclear euchromatin"),
    28: InstanceName(short="Nucleoplasm", long="Nucleoplasm"),
    29: InstanceName(short="Nucleolus", long="Nucleolus"),
    30: InstanceName(short="Microtubules out", long="Microtubules out"),
    31: InstanceName(short="Centrosome", long="Centrosome"),
    32: InstanceName(short="Distal appendages", long="Distal appendages"),
    33: InstanceName(short="Subdistal appendages", long="Subdistal appendages"),
    34: InstanceName(short="Ribosomes", long="Ribsoomes"),
    35: InstanceName(short="Cytosol", long="Cytosol"),
    36: InstanceName(short="Microtubules in", long="Microtubules in"),
    37: InstanceName(short="Nucleus combined", long="Nucleus combined"),
    38: InstanceName(short="Vimentin", long="Vimentin"),
    39: InstanceName(short="Glycogen", long="Glycogen"),
    40: InstanceName(short="Cardiac neurons", long="Cardiac neurons"),
    41: InstanceName(short="Endothelial cells", long="Endothelial cells"),
    42: InstanceName(short="Cardiomyocytes", long="Cardiomyocytes"),
    43: InstanceName(short="Epicardial cells", long="Epicardial cells"),
    44: InstanceName(
        short="Parietal pericardial cells", long="Parietal pericardial cells"
    ),
    45: InstanceName(short="Red blood cells", long="Red blood cells"),
    46: InstanceName(short="White blood cells", long="White blood cells"),
    47: InstanceName(short="Peroxisome membrane", long="Peroxisome membrane"),
    48: InstanceName(short="Peroxisome lumen", long="Peroxisome lumen"),
}

Possibility = Literal["unknown", "absent"]


class SemanticSegmentation(StrictBase):
    type: Literal["semantic_segmentation"] = "semantic_segmentation"
    encoding: Dict[Union[Possibility, Literal["present"]], int]


class InstanceSegmentation(StrictBase):
    type: Literal["instance_segmentation"] = "instance_segmentation"
    encoding: Dict[Possibility, int]


AnnotationType = Union[SemanticSegmentation, InstanceSegmentation]

TName = TypeVar("TName", bound=str)


class AnnotationArrayAttrs(GenericModel, Generic[TName]):
    """
    The metadata for an array of annotated values.
    """

    class_name: TName
    # a mapping from values to frequencies
    histogram: Optional[Dict[Possibility, int]]
    # a mapping from class names to values
    # this is array metadata because labels might disappear during downsampling
    annotation_type: AnnotationType

    @root_validator()
    def check_encoding(cls, values):
        if (typ := values.get("type", False)) and (
            hist := values.get("histogram", False)
        ):
            # check that everything in the histogram is encoded
            assert set(typ.encoding.keys()).issuperset((hist.keys())), "Oh no"

        return values


class AnnotationGroupAttrs(GenericModel, Generic[TName]):
    """
    The metadata for an individual annotated semantic class.
    In a storage hierarchy like zarr or hdf5, this metadata is associated with a
    group-like container that contains a collection of arrays that contain the
    annotation data in a multiscale representation.
    """

    class_name: TName
    annotation_type: AnnotationType


class CropGroupAttrs(GenericModel, Generic[TName]):
    """
    The metadata for all annotations in a single crop.
    """

    name: Optional[str]
    description: Optional[str]
    created_by: list[str]
    created_with: list[str]
    start_date: Optional[date]
    end_date: Optional[date]
    duration_days: Optional[int]
    protocol_url: Optional[str]
    class_names: list[TName]
    index: dict[TName, str]
    doi: Optional[str]


AnnotationArray = ArraySpec[AnnotationArrayAttrs]
AnnotationGroup = GroupSpec[
    CellmapWrapper[AnnotationWrapper[AnnotationGroupAttrs]], AnnotationArray
]
CropGroup = GroupSpec[
    CellmapWrapper[AnnotationWrapper[CropGroupAttrs]], AnnotationGroup
]


def annotation_attrs_wrapper(
    value: T,
) -> dict[Literal["cellmap"], dict[Literal["annotation"], T]]:
    return {"cellmap": {"annotation": value}}


def annotation_array_metadata():
    pass


def annotation_group_metadata():
    pass
