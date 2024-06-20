from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    from xarray import DataArray

import pytest
from cellmap_schemas.multiscale.neuroglancer_n5 import Group
from zarr import N5FSStore

from fibsem_tools.coordinate import stt_from_array
from fibsem_tools.io.n5.hierarchy.neuroglancer import create_dataarray
from tests.conftest import PyramidRequest


@pytest.mark.parametrize(
    "pyramid",
    [
        PyramidRequest(
            dims=("z", "y", "x"),
            shape=(12, 13, 14),
            scale=(1, 2, 3),
            translate=(0, 0, 0),
        ),
        PyramidRequest(
            dims=("z", "y", "x"),
            shape=(22, 53, 14),
            scale=(4, 6, 3),
            translate=(0, 0, 0),
        ),
    ],
    indirect=["pyramid"],
)
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("chunks", ["auto", (10, 10, 10)])
def test_read_array(
    pyramid: tuple[DataArray, DataArray, DataArray],
    use_dask: bool,
    chunks: Literal["auto"] | tuple[int, int, int],
    tmp_n5: str,
):
    """
    Test that we can read an n5 dataset that uses neuroglancer-compatible saalfeld lab metadata
    """
    # insert attributes that should appear on the other end
    [p.attrs.update({"foo": 10}) for p in pyramid]
    paths = ("s0", "s1", "s2")
    store = N5FSStore(tmp_n5)
    transforms = tuple(stt_from_array(array) for array in pyramid)
    group_model = Group.from_arrays(
        arrays=pyramid,
        chunks=(10, 10, 10),
        paths=paths,
        scales=[t.scale for t in transforms],
        axes=transforms[0].axes,
        units=transforms[0].units,
        dimension_order="C",
    )
    group = group_model.to_zarr(store=store, path="pyramid")
    if not use_dask and chunks != "auto":
        msg = f"If use_dask is False, then chunks must be 'auto'. Got {chunks} instead."
        with pytest.raises(ValueError, match=re.escape(msg)):
            observed = tuple(
                create_dataarray(array=group[path], use_dask=use_dask, chunks=chunks)
                for path in paths
            )
        return None
    else:
        observed = tuple(
            create_dataarray(array=group[path], use_dask=use_dask, chunks=chunks)
            for path in paths
        )
        result = tuple(a.equals(b) for a, b in zip(observed, pyramid))
        assert result == (True, True, True)
