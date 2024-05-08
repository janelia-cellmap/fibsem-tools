import pytest
from xarray import DataArray
from fibsem_tools.metadata.neuroglancer import read_dataarray
from fibsem_tools.metadata.transform import stt_from_array
from tests.conftest import PyramidRequest
from zarr import N5FSStore
from cellmap_schemas.multiscale.neuroglancer_n5 import Group


@pytest.mark.parametrize(
    "pyramid",
    (
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
    ),
    indirect=["pyramid"],
)
def test_read_array(pyramid: tuple[DataArray, DataArray, DataArray], tmp_n5: str):
    """
    Test that we can read an n5 dataset that uses neuroglancer-compatible saalfeld lab metadata
    """
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
    observed = tuple(read_dataarray(array=group[path]) for path in paths)
    result = tuple(a.equals(b) for a, b in zip(observed, pyramid))
    assert result == (True, True, True)
