import tempfile
import shutil
import pytest
import atexit


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, path)
    return path


@pytest.fixture
def temp_zarr():
    path = tempfile.mkdtemp(suffix=".zarr")
    atexit.register(shutil.rmtree, path)
    return path


@pytest.fixture
def temp_n5():
    path = tempfile.mkdtemp(suffix=".n5")
    atexit.register(shutil.rmtree, path)
    return path
