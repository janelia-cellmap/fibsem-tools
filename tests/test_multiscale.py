from typing import Tuple
import pytest
import dask.array as da
from xarray import DataArray
from fibsem_tools.io.core import access, read
from fibsem_tools.io.dask import store_blocks
from numcodecs import GZip
from fibsem_tools.io.multiscale import multiscale_group
import zarr


@pytest.mark.parametrize(
    "metadata_types",
    [("ome-ngff@0.4",), ("neuroglancer",), ("ome-ngff",), ("ome-ngff", "neuroglancer")],
)
def test_multiscale_storage(temp_zarr, metadata_types: Tuple[str, ...]):
    data = da.random.randint(0, 8, (16, 16, 16), chunks=(8, 8, 8), dtype="uint8")
    store = zarr.NestedDirectoryStore(temp_zarr)
    coords = [
        DataArray(
            da.arange(data.shape[0]) + 10,
            attrs={"units": "nm", "type": "space"},
            dims=("z",),
        ),
        DataArray(
            da.arange(data.shape[1]) + 20,
            attrs={"units": "nm", "type": "space"},
            dims=("y"),
        ),
        DataArray(
            da.arange(data.shape[2]) - 30,
            attrs={"units": "nm", "type": "space"},
            dims=("x",),
        ),
    ]
    multi = [DataArray(data, coords=coords)]
    multi.append(multi[0].coarsen({"x": 2, "y": 2, "z": 2}).mean().astype("uint8"))
    array_paths = ["s0", "s1"]
    chunks = (8, 8, 8)
    g_spec = multiscale_group(
        multi,
        metadata_types=metadata_types,
        array_paths=array_paths,
        chunks=chunks,
        compressor=GZip(-1),
    )

    multi = [m.chunk(c) for m, c in zip(multi, chunks)]

    group = g_spec.to_zarr(store, path="/")

    array_urls = [f"{temp_zarr}/{ap}" for ap in array_paths]
    da.compute(store_blocks(multi, [access(a_url, mode="a") for a_url in array_urls]))

    assert dict(group.attrs) == g_spec.attrs
    assert all(read(a).chunks == chunks for a in array_urls)
