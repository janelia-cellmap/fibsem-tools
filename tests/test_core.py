import os

import pytest
import zarr
from fibsem_tools.chunk import normalize_chunks
from fibsem_tools.io.core import access, model_multiscale_group, split_by_suffix
from numcodecs import GZip
from xarray import DataArray

from tests.conftest import PyramidRequest


def test_path_splitting():
    path = "s3://0/1/2.n5/3/4"
    split = split_by_suffix(path, (".n5",))
    assert split == ("s3://0/1/2.n5", "3/4", ".n5")

    path = os.path.join("0", "1", "2.n5", "3", "4")
    split = split_by_suffix(path, (".n5",))
    assert split == (os.path.join("0", "1", "2.n5"), os.path.join("3", "4"), ".n5")

    path = os.path.join("0", "1", "2.n5")
    split = split_by_suffix(path, (".n5",))
    assert split == (os.path.join("0", "1", "2.n5"), "", ".n5")


@pytest.mark.parametrize(
    "pyramid",
    (PyramidRequest(shape=(12, 13, 14), scale=(1, 2, 3), translate=(4, 5, 6)),),
    indirect=["pyramid"],
)
@pytest.mark.parametrize(
    "metadata_type",
    ["ome-ngff@0.4", "neuroglancer", "ome-ngff"],
)
def test_multiscale_storage(
    pyramid: tuple[DataArray, DataArray, DataArray],
    tmp_zarr: str,
    metadata_type: str,
) -> None:
    if metadata_type == "neuroglancer":
        store = zarr.N5FSStore(tmp_zarr)
    else:
        store = zarr.NestedDirectoryStore(tmp_zarr)

    array_paths = ["s0", "s1", "s2"]
    pyr = dict(zip(array_paths, pyramid))
    chunks = (8,) * 3
    _chunks = normalize_chunks(pyramid, chunks)
    g_spec = model_multiscale_group(
        arrays=pyr,
        metadata_type=metadata_type,
        chunks=_chunks,
        compressor=GZip(-1),
    )

    group = g_spec.to_zarr(store, path="/")

    array_urls = [f"{tmp_zarr}/{ap}" for ap in array_paths]
    for a_url, d in zip(array_urls, pyr.values()):
        access(a_url, mode="a")[:] = d.data

    assert group.attrs.asdict() == g_spec.attributes.model_dump()
    assert all(a.chunks == chunks for name, a in group.arrays())
