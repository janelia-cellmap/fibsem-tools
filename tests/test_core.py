import os
from typing import Literal

import numpy as np
import pytest
import zarr
from fibsem_tools.chunk import normalize_chunks
from fibsem_tools.io.core import access, model_multiscale_group, split_by_suffix
from numcodecs import GZip
from pydantic_zarr.v2 import ArraySpec, GroupSpec
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


@pytest.mark.parametrize("fmt", ("zarr", "n5"))
@pytest.mark.parametrize("mode", ("r", "a", "w"))
def test_access_zarr_n5(
    tmpdir, fmt: Literal["zarr", "n5"], mode: Literal["r", "a", "w"]
):
    group_path = "group"
    array_path = "array"
    if fmt == "zarr":
        store_path = os.path.join(str(tmpdir), "test.zarr")
        store = zarr.NestedDirectoryStore(store_path)
        dimsep = "/"
    elif fmt == "n5":
        store_path = os.path.join(str(tmpdir), "test.n5")
        store = zarr.N5FSStore(store_path)
        dimsep = "."

    model = GroupSpec(
        attributes={"foo": 10},
        members={
            array_path: ArraySpec.from_array(np.arange(10), dimension_separator=dimsep)
        },
    )
    stored = model.to_zarr(store=store, path=group_path)
    if mode in ("r", "a"):
        assert (
            model.members[array_path].shape
            == access(os.path.join(store_path, group_path, array_path), mode=mode).shape
        )
        assert (
            model.attributes
            == access(os.path.join(store_path, group_path), mode=mode).attrs.asdict()
        )
    else:
        assert isinstance(
            access(os.path.join(store_path, group_path, array_path), mode=mode),
            zarr.Group,
        )


@pytest.mark.parametrize(
    "pyramid",
    (PyramidRequest(shape=(12, 13, 14), scale=(1, 2, 3), translate=(0, 0, 0)),),
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

    assert group.attrs.asdict() == g_spec.attributes.model_dump()
    assert all(a.chunks == chunks for name, a in group.arrays())
