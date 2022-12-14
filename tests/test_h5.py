from h5py._hl.dataset import make_new_dset
from fibsem_tools.io.h5 import partition_h5_kwargs
from inspect import signature, Parameter


def test_kwarg_partition():
    dataset_creation_sig = signature(make_new_dset)
    dataset_kwargs = {
        k: None
        for k, v in filter(
            lambda p: p[1].default is not Parameter.empty,
            dataset_creation_sig.parameters.items(),
        )
    }
    file_kwargs = {"foo": None, "bar": None}
    file_kwargs_out, dataset_kwargs_out = partition_h5_kwargs(
        **dataset_kwargs, **file_kwargs
    )
    assert file_kwargs == file_kwargs_out
    assert dataset_kwargs == dataset_kwargs_out
