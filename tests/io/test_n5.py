from __future__ import annotations

from typing import TYPE_CHECKING

from fibsem_tools.coordinate import stt_from_array
from fibsem_tools.io.n5.core import to_dask
from tests.conftest import PyramidRequest

if TYPE_CHECKING:
    from typing import Literal

    from xarray import DataArray

import dask.array as da
import numpy as np
import pytest
import zarr

from fibsem_tools.io.n5.core import access
from fibsem_tools.io.n5.hierarchy import cosem, neuroglancer


def test_access_array(tmp_n5: str) -> None:
    data = np.random.randint(0, 255, size=(100,), dtype="uint8")
    z = zarr.open(tmp_n5, mode="w", shape=data.shape, chunks=10)
    z[:] = data
    assert np.array_equal(access(tmp_n5, "", mode="r")[:], data)


def test_access_group(tmp_zarr: str) -> None:
    data = np.zeros(100, dtype="uint8") + 42
    path = "foo"
    zg = zarr.open(zarr.N5FSStore(tmp_zarr), mode="a")
    zg[path] = data
    zg.attrs["bar"] = 10
    assert access(tmp_zarr, "", mode="a") == zg

    zg = access(tmp_zarr, "", mode="w", attrs={"bar": 10})
    zg["foo"] = data
    assert zarr.open(zarr.N5FSStore(tmp_zarr), mode="a") == zg


@pytest.mark.parametrize("chunks", ["auto", (10,)])
def test_dask(tmp_n5: str, chunks: Literal["auto"] | tuple[int, ...]) -> None:
    path = "foo"
    data = np.arange(100)
    zarray = access(tmp_n5, path, mode="w", shape=data.shape, dtype=data.dtype)
    zarray[:] = data
    name_expected = "foo"

    expected = da.from_array(zarray, chunks=chunks, name=name_expected)
    observed = to_dask(zarray, chunks=chunks, name=name_expected)

    assert observed.chunks == expected.chunks
    assert observed.name == expected.name
    assert np.array_equal(observed, data)


@pytest.mark.parametrize("metadata_type", ["neuroglancer", "cosem"])
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
            translate=(3, 4, 6),
        ),
    ],
    indirect=["pyramid"],
)
@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("name", [None, "foo"])
@pytest.mark.parametrize("chunks", [(5, 5, 5)])
def test_read_dataarray(
    tmpdir,
    metadata_type: Literal["neuroglancer", "cosem"],
    pyramid: list[DataArray],
    use_dask: bool,
    name: str | None,
    chunks: tuple[int, int, int],
) -> None:
    array_names = ("s0", "s1", "s2")
    pyramid_dict = dict(zip(array_names, pyramid))
    path = "test"

    base_tx = stt_from_array(pyramid[0])
    nonzero_translate = any(v != 0 for v in base_tx.translate)
    store = zarr.N5FSStore(str(tmpdir))
    if metadata_type == "cosem":
        group_model = cosem.model_group(arrays=pyramid_dict, chunks=chunks)
        dataarray_creator = cosem.create_dataarray
    elif metadata_type == "neuroglancer":
        if nonzero_translate:
            match = (
                "Be advised that this translation parameter will not be stored, due to limitations "
                "of the metadata format you are using."
            )
            with pytest.warns(UserWarning, match=match):
                group_model = neuroglancer.model_group(
                    arrays=pyramid_dict, chunks=chunks
                )
        else:
            group_model = neuroglancer.model_group(arrays=pyramid_dict, chunks=chunks)
        dataarray_creator = neuroglancer.create_dataarray
    else:
        msg = f"Metadata format {metadata_type} not recognized"
        raise ValueError(msg)

    group = group_model.to_zarr(store, path=path)

    for name, value in pyramid_dict.items():
        observed = dataarray_creator(
            group[name],
            use_dask=use_dask,
        )
        assert observed.dims == value.dims
        if not nonzero_translate:
            assert all(
                a.equals(b)
                for a, b in zip(observed.coords.values(), value.coords.values())
            )
        assert isinstance(observed.data, da.Array) == use_dask
