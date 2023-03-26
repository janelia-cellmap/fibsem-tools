from zarr.storage import FSStore
import zarr
import numpy as np
from fibsem_tools.io.zarr import get_url


def test_url(temp_zarr):
    store = FSStore(temp_zarr)
    group = zarr.group(store)
    arr = group.create_dataset(name="foo", data=np.arange(10))
    assert get_url(arr) == f"file://{store.path}/foo"
