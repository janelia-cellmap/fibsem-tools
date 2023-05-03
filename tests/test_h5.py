import os
import numpy as np
from fibsem_tools.io.h5 import access_h5
import h5py


def test_access_array(temp_dir):
    path = os.path.join(temp_dir, "foo.h5")
    key = "s0"
    data = np.random.randint(0, 255, size=(10, 10, 10), dtype="uint8")
    attrs = {"resolution": "1000"}

    with h5py.File(path, mode="w") as h5f:
        arr1 = h5f.create_dataset(key, data=data)
        arr1.attrs.update(**attrs)

    arr2 = access_h5(path, key, mode="r")
    assert dict(arr2.attrs) == attrs
    assert np.array_equal(arr2[:], data)
    arr2.file.close()

    arr3 = access_h5(path, key, data=data, attrs=attrs, mode="w")
    assert dict(arr3.attrs) == attrs
    assert np.array_equal(arr3[:], data)
    arr3.file.close()


def test_access_group(temp_dir):
    key = "s0"
    store = os.path.join(temp_dir, key)
    attrs = {"resolution": "1000"}

    grp = access_h5(store, key, attrs=attrs, mode="w")
    assert dict(grp.attrs) == attrs
    grp.file.close()

    grp2 = access_h5(store, key, mode="r")
    assert dict(grp2.attrs) == attrs
    grp2.file.close()
