from numpy.random.mtrand import f
from fibsem_tools.io.multiscale import Multiscales
from xarray import DataArray
import shutil
import dask.array as da
import dask
import tempfile
import atexit
import numpy as np
import pytest
from fibsem_tools.metadata.cosem import COSEMGroupMetadata
from fibsem_tools.metadata.neuroglancer import NeuroglancerN5GroupMetadata
from fibsem_tools.metadata.transform import SpatialTransform


@pytest.mark.parametrize("multiscale_metadata", (True, False))
@pytest.mark.parametrize("propagate_array_attrs", (True, False))
def test_multiscale_storage(multiscale_metadata: bool, propagate_array_attrs: bool):
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
    )
    dask.delayed(storage).compute()

    for key, value in ms.attrs.items():
        assert group.attrs[key] == value

    for idx, k in enumerate(multi):
        assert group[k] == arrays[idx]
        assert arrays[idx].shape == multi[k].shape
        assert arrays[idx].dtype == multi[k].dtype
        assert arrays[idx].chunks == multi[k].data.chunksize
        assert np.array_equal(arrays[idx][:], multi[k].data.compute())
        if propagate_array_attrs:
            assert arrays[idx].attrs[f"{k}/foo"] == multi[k].attrs[f"{k}/foo"]
        else:
            assert f"{k}/foo" not in arrays[idx].attrs

        if multiscale_metadata:
            cosem_meta = COSEMGroupMetadata.fromDataArrays(
                name=ms.name,
                paths=tuple(multi.keys()),
                dataarrays=tuple(multi.values()),
            ).dict()
            for mkey, mvalue in cosem_meta.items():
                assert group.attrs[mkey] == mvalue

            neuroglancer_meta = NeuroglancerN5GroupMetadata.fromDataArrays(
                dataarrays=tuple(multi.values())
            ).dict()
            for mkey, mvalue in neuroglancer_meta.items():
                assert group.attrs[mkey] == mvalue

            assert (
                arrays[idx].attrs["transform"]
                == SpatialTransform.fromDataArray(dataarray=multi[k]).dict()
            )
        else:
            assert "multiscales" not in group.attrs
