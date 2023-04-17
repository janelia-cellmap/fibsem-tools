from importlib.metadata import version as _version

__version__ = _version(__name__)

# ruff: noqa
from fibsem_tools.io.core import read, read_dask, read_xarray, access, create_group
from fibsem_tools._dataset import CosemDataset

__all__ = [
    "access",
    "create_group",
    "read",
    "read_dask",
    "read_xarray",
    "CosemDataset",
]
