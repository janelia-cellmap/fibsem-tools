import datetime
from typing import Literal, TYPE_CHECKING
from pydantic import BaseModel
from functools import lru_cache
import supabase.client

if TYPE_CHECKING:
    from postgrest import SyncSelectRequestBuilder, SyncRequestBuilder

ArrayContainerFormat = Literal["n5", "zarr", "precomputed"]
SampleType = Literal["scalar", "label"]
ContentType = Literal["lm", "prediction", "segmentation", "em", "analysis"]
PublicationType = Literal["paper", "doi"]
MeshFormat = Literal["neuroglancer_multilod_draco", "neuroglancer_legacy_mesh"]
CountMethod = Literal["exact", "planned", "estimated"]

SUPA_URL = "https://kvwjcggnkipomjlbjykf.supabase.co"
SUPA_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt2d2pjZ2dua2lwb21qbGJqeWtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjUxODgyMjksImV4cCI6MTk4MDc2NDIyOX0.o_yLKX9erKbIrG3mwdwFkWYI8N9EjTNUnu9FWMngw9E"  # noqa: E501


@lru_cache(maxsize=1)
def get_client(key: str = SUPA_KEY, url: str = SUPA_URL) -> supabase.client.Client:
    return supabase.client.create_client(url, key)


class SupaModel(BaseModel):
    @classmethod
    def select(
        cls, *columns: str, count: CountMethod | None = None
    ) -> "SyncSelectRequestBuilder":
        return cls.table().select(*columns, count=count)

    @classmethod
    def table(cls) -> "SyncRequestBuilder":
        table_name = cls.__name__.lower()
        return get_client().table(table_name)


class Sample(SupaModel):
    id: int | None = None
    name: str | None = None
    description: str
    protocol: str
    contributions: str
    organism: list[str] | None
    type: list[str] | None
    subtype: list[str] | None
    treatment: list[str] | None


class Dataset(SupaModel):
    id: int
    name: str
    description: str
    thumbnail_url: str
    is_published: bool
    sample: Sample
    created_at: str
    acquisition_id: int


class Transform(SupaModel):
    axes: list[str] | None
    scale: list[float] | None
    order: str | None = None
    units: list[str] | None = None
    translate: list[float] | None = None


class ContrastLimits(SupaModel):
    start: int
    end: int
    min: int
    max: int


class DisplaySettings(SupaModel):
    color: str | None = None
    invertLUT: bool = False
    contrastLimits: ContrastLimits


class Image(SupaModel):
    id: int
    name: str
    description: str
    url: str
    format: ArrayContainerFormat
    transform: Transform
    display_settings: DisplaySettings
    created_at: datetime.datetime
    sample_type: SampleType
    content_type: ContentType
    dataset_name: str
    institution: str


class ImageAcquisition(SupaModel):
    id: int
    name: str
    institution: str
    start_date: datetime.datetime
    grid_axes: list[str]
    grid_spacing: list[float]
    grid_spacing_unit: str
    grid_dimensions: list[float]
    grid_dimensions_unit: str


class Mesh(SupaModel):
    id: int
    name: str
    description: str
    created_at: datetime.datetime
    url: str
    transform: Transform
    image_id: int
    format: MeshFormat
    ids: list[int]


class Publication(SupaModel):
    id: int
    name: str
    url: str
    type: PublicationType


class PublicationToDataset(SupaModel):
    dataset_name: str
    publication_id: int


class Taxon(SupaModel):
    id: int
    created_at: datetime.datetime | None
    name: str
    short_name: str


class View(SupaModel):
    id: int
    name: str
    description: str
    created_at: datetime.datetime
    position: list[float] | None
    scale: float | None
    orientation: list[float] | None
    # tags: list[str] | None
    dataset_name: str
    thumbnail_url: str | None


class ViewToImage(SupaModel):
    image_id: int
    view_id: int


class ViewToTaxon(SupaModel):
    taxon_id: int
    view_id: int
