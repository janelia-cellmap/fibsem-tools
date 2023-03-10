import atexit
import shutil
import tempfile
from typing import Optional, Tuple

import dask
import dask.array as da
import numpy as np
import pytest
from xarray import DataArray
from fibsem_tools.io.core import read

from fibsem_tools.io.multiscale import (
    Multiscales,
    multiscale_group,
    multiscale_metadata,
)
from fibsem_tools.metadata.cosem import COSEMGroupMetadata
from fibsem_tools.metadata.neuroglancer import NeuroglancerN5GroupMetadata
from fibsem_tools.metadata.transform import STTransform


@pytest.mark.parametrize("chunks", (None, (2, 2)))
@pytest.mark.parametrize("multiscale_metadata", (True, False))
@pytest.mark.parametrize("propagate_array_attrs", (True, False))
def test_multiscale_storage(
    multiscale_metadata: bool,
    propagate_array_attrs: bool,
    chunks: Optional[Tuple[int, int]],
):
    data = da.random.randint(0, 8, (16, 16), chunks=(8, 8), dtype="uint8")
    coords = (
        ("x", da.arange(data.shape[0]), {"units": "nm"}),
        ("y", da.arange(data.shape[1]), {"units": "nm"}),
    )
    multi = {"s0": DataArray(data, coords=coords)}
    multi["s1"] = multi["s0"].coarsen({"x": 2, "y": 2}).mean()
    for k in multi:
        multi[k].attrs[f"{k}/foo"] = f"{k}/bar"

    ms = Multiscales("test", multi, attrs={"foo": "bar"})

    store = tempfile.mkdtemp(suffix=".zarr")
    atexit.register(shutil.rmtree, store)

    group, arrays, storage = ms.store(
        store,
        multiscale_metadata=multiscale_metadata,
        propagate_array_attrs=propagate_array_attrs,
        write_empty_chunks=False,
        chunks=chunks,
    )
    assert all([a.write_empty_chunks is False for a in arrays])
    dask.delayed(storage).compute()

    for key, value in ms.attrs.items():
        assert group.attrs[key] == value

    for idx, k in enumerate(multi):
        assert group[k] == arrays[idx]
        assert arrays[idx].shape == multi[k].shape
        assert arrays[idx].dtype == multi[k].dtype

        if chunks is None:
            assert arrays[idx].chunks == multi[k].data.chunksize
        else:
            assert arrays[idx].chunks == chunks
        assert np.array_equal(arrays[idx][:], multi[k].data.compute())
        if propagate_array_attrs:
            assert arrays[idx].attrs[f"{k}/foo"] == multi[k].attrs[f"{k}/foo"]
        else:
            assert f"{k}/foo" not in arrays[idx].attrs

        if multiscale_metadata:
            cosem_meta = COSEMGroupMetadata.fromDataArrays(
                name=ms.name,
                paths=tuple(multi.keys()),
                arrays=tuple(multi.values()),
            ).dict()
            for mkey, mvalue in cosem_meta.items():
                assert group.attrs[mkey] == mvalue

            neuroglancer_meta = NeuroglancerN5GroupMetadata.fromDataArrays(
                arrays=tuple(multi.values())
            ).dict()
            for mkey, mvalue in neuroglancer_meta.items():
                assert group.attrs[mkey] == mvalue

            assert (
                arrays[idx].attrs["transform"]
                == STTransform.fromDataArray(array=multi[k]).dict()
            )
        else:
            assert "multiscales" not in group.attrs


def test_multiscale_storage_2():
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
    multi[1] = multi[1].assign_coords(
        {dim: multi[1].coords[dim] - 0.5 for dim in multi[1].dims}
    )
    array_paths = ["s0", "s1"]
    g_meta, a_meta = multiscale_metadata(
        multi, metadata_types=["ome-ngff"], array_paths=array_paths
    )

    chunks = ((8, 8, 8), (8, 8, 8))

    group_url, array_urls = multiscale_group(
        store,
        multi,
        array_paths=array_paths,
        metadata_types=["ome-ngff"],
        chunks=chunks,
    )

    assert dict(read(group_url).attrs) == g_meta
    assert tuple(read(a).chunks for a in array_urls) == chunks
