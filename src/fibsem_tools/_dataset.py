from __future__ import annotations

import json
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence
from urllib.request import urlopen


if TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import TypedDict
    import xarray as xr

    class DatasetMetadata(TypedDict):
        title: str
        id: str
        imaging: dict
        sample: dict
        institution: list[str]
        softwareAvailability: str
        DOI: list[dict]
        publications: list[dict]

    class DatasetView(TypedDict):
        """
        - sources: suggested layers
        - position: [X, Y, Z] centerpoint of the feature
        - scale: nm/pixel at which to show the view
        - orientation: always seems to be [1, 0, 0, 0].
        """

        name: str
        description: str
        sources: list[str]
        position: list[float] | None
        scale: float | None
        orientation: list[float]

    class Source(TypedDict):
        name: str
        description: str
        url: str
        format: str
        transform: dict
        sampleType: str
        contentType: str
        displaySettings: dict
        subsources: list

    class DatasetManifest(TypedDict):
        name: str
        metadata: DatasetMetadata
        sources: dict[str, Source]
        views: list[DatasetView]


GH_API = "https://raw.githubusercontent.com/janelia-cosem/fibsem-metadata/stable/api"
COSEM_S3 = "s3://janelia-cosem-datasets"


class CosemDataset:
    def __init__(self, id: str) -> None:
        self.id = id

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def title(self) -> str:
        return self.metadata["title"]

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return (
            f"<CosemDataset '{self}' sources: {len(self.sources)}, "
            f"views: {len(self.views)}>"
        )

    @property
    def manifest(self) -> DatasetManifest:
        return get_manifest(self.id)

    @property
    def metadata(self) -> DatasetMetadata:
        return self.manifest["metadata"]

    # @property
    # def thumbnail(self) -> np.ndarray:
    #     return get_thumbnail(self.id)

    @property
    def views(self) -> list[DatasetView]:
        return self.manifest["views"]

    @property
    def sources(self) -> dict[str, Source]:
        return self.manifest["sources"]

    def read_source(self, key: str, level: int = 0) -> xr.DataArray:
        import fibsem_tools

        source = self.sources[key]
        if source["format"] != "n5":  # pragma: no cover
            raise NotImplementedError(
                f"Can only read n5 sources, (not {source['format']!r})"
            )

        return fibsem_tools.read_xarray(  # type: ignore [attr-defined]
            f"{source['url']}/s{level}", storage_options={"anon": True}
        )

    def view(self, name: str) -> DatasetView:
        for d in self.views:
            if d["name"].lower().startswith(name.lower()):
                return d
        raise KeyError(f"No view named/starting with {name!r}")  # pragma: no cover

    def load_view(
        self,
        name: str | None = None,
        sources: Sequence[str] = (),
        position: Sequence[float] | None = None,
        exclude: set[str] | None = None,
        extent: float | Sequence[float] | None = 1000,  # in nm around position
        level: int = 0,
    ) -> xr.DataArray:
        return load_view(
            self,
            name=name,
            sources=sources,
            position=position,
            exclude=exclude,
            extent=extent,
            level=level,
        )

    @staticmethod
    def all_names() -> list[str]:
        return list(get_datasets())


@lru_cache
def get_datasets() -> dict[str, str]:
    """Retrieve available datasets from janelia-cosem/fibsem-metadata."""
    with urlopen(f"{GH_API}/index.json") as r:  # noqa: S310
        return json.load(r).get("datasets")


@lru_cache(maxsize=64)
def get_manifest(dataset: str) -> DatasetManifest:
    """Get manifest for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset ID, e.g. "jrc_hela-3".

    Returns
    -------
    Dict[str, str]
        Useful keys include:

        * views: a curated list of views with:

    """
    with urlopen(f"{GH_API}/{dataset}/manifest.json") as r:  # noqa: S310
        return json.load(r)


def load_view(
    dataset: str | CosemDataset,
    sources: Sequence[str] = (),
    name: str | None = None,
    exclude: set[str] | None = None,
    extent: float | Sequence[float] | None = None,  # in nm around position, in XYZ
    position: Sequence[float] | None = None,  # in XYZ
    level: int = 0,
) -> xr.DataArray:
    import dask
    import xarray as xr

    ds = CosemDataset(dataset) if isinstance(dataset, str) else dataset
    if name is not None:
        view = ds.view(name)
        position = view["position"]
        if not sources:
            sources = view["sources"]

    if exclude is None:
        exclude = set()

    sources = [
        s.replace("fibsem-uint8", "fibsem-uint16")
        for s in sources
        if ds.sources.get(s, {}).get("contentType") not in exclude
    ]
    _loaded: list[str] = []
    arrs: list[xr.DataArray] = []
    for source in sources:
        try:
            arrs.append(ds.read_source(source, level))
            _loaded.append(source)
        except (NotImplementedError, KeyError):  # pragma: no cover
            warnings.warn(f"Could not load source {source!r}", stacklevel=2)

    if not arrs:  # pragma: no cover
        raise RuntimeError("No sources could be loaded")

    if len(arrs) > 1:
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
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
