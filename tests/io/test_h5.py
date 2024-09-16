from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.compat import LEGACY_PATH

import os
import pytest
import h5py
import numpy as np

from fibsem_tools.io.h5 import access


@pytest.mark.parametrize("key", ("s0", "s2"))
def test_access_array(tmpdir: LEGACY_PATH, key: str) -> None:
    path = os.path.join(str(tmpdir), "foo.h5")
    data = np.random.randint(0, 255, size=(10, 10, 10), dtype="uint8")
    attrs = {"resolution": "1000"}

    with h5py.File(path, mode="w") as h5f:
        arr1 = h5f.create_dataset(key, data=data)
        arr1.attrs.update(**attrs)

    arr2 = access(path, key, mode="r")
    assert dict(arr2.attrs) == attrs
    assert np.array_equal(arr2[:], data)
    arr2.file.close()

    arr3 = access(path, key, data=data, attrs=attrs, mode="w")
    assert dict(arr3.attrs) == attrs
    assert np.array_equal(arr3[:], data)
    arr3.file.close()


@pytest.mark.parametrize("key", ("a", "/", ""))
def test_access_group(tmpdir: LEGACY_PATH, key: str) -> None:
    store = os.path.join(str(tmpdir), "test.h5")
    attrs = {"resolution": "1000"}

    grp = access(store, key, attrs=attrs, mode="w")
    assert dict(grp.attrs) == attrs
    grp.file.close()

    grp2 = access(store, key, mode="r")
    assert dict(grp2.attrs) == attrs
    grp2.file.close()
