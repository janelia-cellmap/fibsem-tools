from pydantic import BaseModel
from typing import Dict, List, Sequence, Tuple, TypeAlias, Union, Literal, Optional

from xarray import DataArray

JSON: TypeAlias = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]

OmeNgffVersion = "0.5-dev"

SpaceUnit = Literal[
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]

TimeUnit = Literal[
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]

AxisType = Literal["space", "time", "channel"]


class StrictBaseModel(BaseModel):
    class config:
        extra = "forbid"


class PathTransform(StrictBaseModel):
    type: Union[Literal["scale"], Literal["translation"]]
    path: str


class VectorTranslationTransform(StrictBaseModel):
    type: Literal["translation"] = "translation"
    translation: List[float]


class VectorScaleTransform(StrictBaseModel):
    type: Literal["scale"] = "scale"
    scale: List[float]


ScaleTransform: TypeAlias = Union[VectorScaleTransform, PathTransform]
TranslationTransform: TypeAlias = Union[VectorTranslationTransform, PathTransform]


class Axis(StrictBaseModel):
    name: str
    type: Optional[Union[AxisType, str]]
    unit: Optional[Union[TimeUnit, SpaceUnit, str]]


class MultiscaleDataset(BaseModel):
    path: str
    coordinateTransformations: Union[
        Tuple[ScaleTransform], Tuple[ScaleTransform, TranslationTransform]
    ]


class MultiscaleMetadata(BaseModel):
    version: Optional[str] = OmeNgffVersion
    name: Optional[str]
    type: Optional[str]
    metadata: Optional[Dict[str, JSON]]
    axes: List[Axis]
    datasets: List[MultiscaleDataset]
    coordinateTransformations: Optional[
        List[Union[ScaleTransform, TranslationTransform]]
    ]


class GroupMetadata(BaseModel):
    multiscales: List[MultiscaleMetadata]
    coordinateTransformations: List[Union[TranslationTransform, ScaleTransform]]

    @classmethod
    def fromDataArrays(
        cls,
        arrays: Sequence[DataArray],
        name: Optional[str] = None,
        paths: Optional[Sequence[str]] = None,
    ):
        raise NotImplementedError("Not done yet")
        # axes =
        # multiscales = MultiscaleMetadata()
        # return cls(multiscales=multiscales, coordinateTransformations=transforms)


class ArrayMetadata(BaseModel):
    axes: List[Axis]
    coordinateTransformations: List[Union[ScaleTransform, TranslationTransform]]

    @classmethod
    def fromDataArray(cls, array: DataArray) -> "ArrayMetadata":
        """
        Generate a ArrayMetadata from a DataArray.

        Parameters
        ----------

        array: DataArray

        Returns
        -------

        ArrayMetadata

        """
        raise NotImplementedError

    # axis_names = [str(d) for d in array.dims]
    # # default unit is meters
    # axis_units = [array.coords[ax].attrs.get("units", None) for ax in axis_names]
    # axis_types =

    # translate = [float(array.coords[ax][0]) for ax in axes]
    # scale = []
    # for ax in axes:
    #     if len(array.coords[ax]) > 1:
    #         scale_estimate = abs(
    #             float(array.coords[ax][1]) - float(array.coords[ax][0])
    #         )
    #    else:
    #         raise ValueError(
    #             f"Cannot infer scale parameter along dimension {ax} with length {len(array.coords[ax])}"
    #         )
    #     scale.append(scale_estimate)

    # return cls(
    #     axes=axes, units=units, translate=translate, scale=scale
    # )
