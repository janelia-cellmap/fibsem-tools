import atexit
import shutil
import tempfile
from typing import Tuple
import pytest
import dask.array as da
from xarray import DataArray
from fibsem_tools.io.core import read

from fibsem_tools.io.multiscale import (
    multiscale_group,
    multiscale_metadata,
)


@pytest.mark.parametrize(
    "metadata_types",
    [("ome-ngff@0.4",), ("neuroglancer",), ("ome-ngff",), ("ome-ngff", "neuroglancer")],
)
def test_multiscale_storage(metadata_types: Tuple[str, ...]):

    store = tempfile.mkdtemp(suffix=".zarr")
    atexit.register(shutil.rmtree, store)

    data = da.random.randint(0, 8, (16, 16, 16), chunks=(8, 8, 8), dtype="uint8")
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
    g_meta, a_meta = multiscale_metadata(
        multi, metadata_types=metadata_types, array_paths=array_paths
    )

    chunks = ((8, 8, 8), (8, 8, 8))

    group_url, array_urls = multiscale_group(
        store,
        multi,
        array_paths=array_paths,
        metadata_types=metadata_types,
        chunks=chunks,
    )

    assert dict(read(group_url).attrs) == g_meta
    assert tuple(read(a).chunks for a in array_urls) == chunks
