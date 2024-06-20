import pytest
from fibsem_tools.type import GroupLike
from pydantic_zarr.v2 import GroupSpec
from zarr import MemoryStore


def test_grouplike() -> None:
    pytest.skip(reason="Zarr is not typed well enough for this to work yet.")
    group = GroupSpec().to_zarr(store=MemoryStore(), path="")
    assert isinstance(group, GroupLike)
