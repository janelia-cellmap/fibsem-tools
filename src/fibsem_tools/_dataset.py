from __future__ import annotations
from collections import defaultdict

import os
import dask
import xarray as xr
import warnings
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Iterator,
    Literal,
    Mapping,
    Sequence,
)
from fibsem_tools import read_xarray


if TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import TypedDict
    import supabase.client
    import matplotlib.pyplot as plt
    import numpy as np

    # todo: find a way to generate these types from either postgrest or an
    # API wrapping postgrest
    class ImageDict(TypedDict):
        id: int
        name: str
        description: str
        url: str
        format: Literal["n5", "zarr"]
        transform: dict[str, Any]
        display_settings: dict[str, Any]
        created_at: str
        sample_type: Literal["scalar"] | Literal["label"]
        content_type: Literal["lm"] | Literal["prediction"] | Literal[
            "segmentation"
        ] | Literal["em"] | Literal["analysis"]
        dataset_name: str
        institution: str

    class ViewDict(TypedDict):
        id: int
        name: str
        description: str
        created_at: str
        position: tuple[float, float, float] | None
        scale: float | None
        orientation: tuple[float, float, float, float] | None
        dataset_name: str
        thumbnail_url: str
        images: list[str]

    class SampleDict(TypedDict):
        type: list[str]
        subtype: list[str]
        organism: list[str]
        protocol: str
        treatment: list[str]
        description: str
        contributions: str

    class ImageAcquisitionDict(TypedDict):
        id: int
        name: str
        institution: str
        start_date: str
        grid_axes: list[str]
        grid_spacing: list[float]
        grid_spacing_unit: str
        grid_dimensions: list[float]
        grid_dimensions_unit: str

    class MetadataDict(TypedDict):
        id: int
        name: str
        description: str
        thumbnail_url: str
        is_published: bool
        sample: SampleDict
        created_at: str
        acquisition_id: int
        image_acquisition: ImageAcquisitionDict
        publication: list[dict[str, Any]]
        images: dict[str, ImageDict]
        views: dict[str, ViewDict]


_DEFAULT_SUPA_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt2d2pjZ2dua2lwb21qbGJqeWtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjUxODgyMjksImV4cCI6MTk4MDc2NDIyOX0.o_yLKX9erKbIrG3mwdwFkWYI8N9EjTNUnu9FWMngw9E"  # noqa: E501
SUPA_KEY = os.environ.get("COSEM_SUPABASE_API_KEY", _DEFAULT_SUPA_KEY)
SUPA_URL = "https://kvwjcggnkipomjlbjykf.supabase.co"


@lru_cache(maxsize=None)
def supaclient(key: str = SUPA_KEY, url: str = SUPA_URL) -> supabase.client.Client:
    import supabase.client

    return supabase.client.create_client(url, key)


@lru_cache(maxsize=None)
def metadata(
    dataset_id: str, client: supabase.client.Client | None = None
) -> MetadataDict:
    """Get overview metatadata for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset ID, e.g. "jrc_hela-3".

    Returns
    -------
    dict
        Metadata for the dataset.
    """
    client = client or supaclient()
    q = (
        "*",
        "image:image(*)",
        "view:view(*)",
        "image_acquisition:image_acquisition(*)",
        "publication:publication(*)",
    )
    query = client.table("dataset").select(",".join(q)).eq("name", dataset_id)
    if not (data := query.execute().data):
        raise ValueError(f"Dataset '{dataset_id}' not found")

    data = data[0]
    data["images"] = {d["name"]: d for d in data.pop("image")}
    data["views"] = {d["name"]: d for d in data.pop("view")}
    view2img = _view_images(client)
    if dataset_id in view2img:
        for view in data["views"].values():
            view["images"] = view2img[dataset_id].get(view["name"], [])
    return data


@lru_cache(maxsize=None)
def _view_images(
    client: supabase.client.Client | None = None,
) -> dict[str, dict[str, list[str]]]:
    client = client or supaclient()
    query = client.table("view_to_image").select(
        "view:view(dataset_name,name), image:image(name)"
    )
    # map of {dataset: {view: {image}}}
    out: dict[str, DefaultDict[str, set[str]]] = {}
    for row in query.execute().data:
        dataset_name = row["view"]["dataset_name"]
        ddict = out.setdefault(dataset_name, defaultdict(set))
        ddict[row["view"]["name"]].add(row["image"]["name"])
    return {
        dset: {vname: sorted(x) for vname, x in v.items()} for dset, v in out.items()
    }


class CosemDataset:
    def __init__(
        self, id: str, api_key: str = SUPA_KEY, api_url: str = SUPA_URL
    ) -> None:
        self.id = id
        self.client = supaclient(api_key, api_url)

    @property
    def name(self) -> str:
        return self.metadata["name"]

    @property
    def description(self) -> str:
        return self.metadata["description"]

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return (
            f"<CosemDataset '{self}' sources: {len(self.images)}, "
            f"views: {len(self.views)}>"
        )

    @property
    def metadata(self) -> MetadataDict:
        return metadata(self.id, self.client)

    @property
    def thumbnail(self) -> np.ndarray:
        from imageio.v3 import imread

        return imread(self.metadata["thumbnail_url"])

    @property
    def images(self) -> dict[str, ImageDict]:
        return self.metadata["images"]

    def read_image(self, image_id: str, level: int = 0) -> xr.DataArray:
        source = self.images[image_id]
        if source["format"] not in ("n5", "zarr"):  # pragma: no cover
            raise NotImplementedError(
                f"Can only read n5 or zarr sources, (not {source['format']!r})"
            )

        return read_xarray(f"{source['url']}/s{level}", storage_options={"anon": True})

    @property
    def views(self) -> dict[str, ViewDict]:
        return self.metadata["views"]

    def view(self, name: str) -> ViewDict:
        for view_name, view in self.views.items():
            if view_name.lower().startswith(name.lower()):
                return view
        raise KeyError(f"No view named/starting with {name!r}")  # pragma: no cover

    def load_view(
        self,
        name: str | None = None,
        images: Sequence[str] = (),
        position: Sequence[float] | None = None,
        exclude: set[str] | None = None,
        extent: float | Sequence[float] | None = 1000,  # in nm around position
        level: int = 0,
    ) -> xr.DataArray:
        return load_view(
            self,
            name=name,
            images=images,
            position=position,
            exclude=exclude,
            extent=extent,
            level=level,
        )

    @staticmethod
    def all_names() -> list[str]:
        return dataset_names()

    @staticmethod
    def all_thumbnails(ncols: int = 4, **kwargs) -> plt.Figure:
        import matplotlib.pyplot as plt
        import math
        from imageio.v3 import imread

        client = supaclient()
        query = client.table("dataset").select("name, thumbnail_url")
        data = query.execute().data
        data = [d for d in data if d["thumbnail_url"].replace('"', "")]

        nrows = math.ceil(len(data) / ncols)
        # kwargs.setdefault("figsize", (18, 18))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
        for ax, row in zip(axes.flat, data):
            uri = row["thumbnail_url"].replace('"', "")
            if not uri:
                continue
            ax.imshow(imread(uri))
            ax.set_title(row["name"])
            ax.axis("off")
        fig.set_tight_layout(True)
        return fig


class Datasets(Mapping[str, CosemDataset]):
    table_name = "dataset"

    def __init__(self, client: supabase.client.Client = supaclient()):
        self.client = client

    def __getitem__(self, key: str) -> CosemDataset:
        import postgrest.exceptions

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("name", key)
                .single()
                .execute()
            )
            return query.data
        except postgrest.exceptions.APIError as e:
            raise KeyError from e

    def __iter__(self) -> Iterator[str, None, None]:
        for data in self.client.table(self.table_name).select("name").execute().data:
            yield data["name"]

    def __len__(self) -> int:
        return len(self.client.table(self.table_name).select("name").execute().data)


# consider making this a static method of `Datasets`
@lru_cache
def dataset_names() -> list[str]:
    """Retrieve available dataset names from janelia-cosem."""
    client = supaclient()
    query = client.table("dataset").select("name")
    return sorted(d["name"] for d in query.execute().data)


def load_view(
    dataset: str | CosemDataset,
    images: Sequence[str] = (),
    name: str | None = None,
    exclude: set[str] | None = None,
    extent: float | Sequence[float] | None = None,  # in nm around position, in XYZ
    position: Sequence[float] | None = None,  # in XYZ
    level: int = 0,
) -> xr.DataArray:
    ds = CosemDataset(dataset) if isinstance(dataset, str) else dataset
    if name is not None:
        view = ds.view(name)
        position = view["position"]
        if not images:
            images = view["images"]

    if exclude is None:
        exclude = set()

    images = [
        s.replace("fibsem-uint8", "fibsem-uint16")
        for s in images
        if ds.images.get(s, {}).get("content_type") not in exclude  # type: ignore
    ]
    _loaded: list[str] = []
    arrs: list[xr.DataArray] = []
    for source in images:
        try:
            arrs.append(ds.read_image(source, level))
            _loaded.append(source)
        except (NotImplementedError, KeyError):  # pragma: no cover
            warnings.warn(f"Could not load source {source!r}", stacklevel=2)

    if not arrs:  # pragma: no cover
        raise RuntimeError("No sources could be loaded")

    if len(arrs) > 1:
        with dask.config.set(
            **{"array.slicing.split_large_chunks": False}
        ):  # type: ignore
            stack = xr.concat(arrs, dim="source")
        stack.coords["source"] = _loaded
    else:
        stack = arrs[0]

    if extent is not None:
        if position is None:
            position = [stack.sizes[i] / 2 for i in "xyz"]
        stack = _crop_around(stack, position, extent)

    # .transpose("source", "y", "z", "x")
    return stack


def _crop_around(
    ary: xr.DataArray,
    position: Sequence[float],
    extent: float | Sequence[float],
    axes: str = "xyz",
) -> xr.DataArray:
    """Crop dataarray around position."""
    if len(position) != 3:  # pragma: no cover
        raise ValueError("position must be of length 3 (X, Y, Z)")
    if isinstance(extent, (float, int)):
        extent = (extent,) * 3
    if len(extent) != 3:  # pragma: no cover
        raise ValueError("extent must be of length 3")

    slc = {ax: slice(p - e / 2, p + e / 2) for p, e, ax in zip(position, extent, axes)}
    return ary.sel(**slc)  # type: ignore
