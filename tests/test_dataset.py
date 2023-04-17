from fibsem_tools._dataset import CosemDataset
import urllib.error
import pytest
import xarray as xr


def test_dataset():
    try:
        name = 'jrc_hela-3'
        assert name in CosemDataset.all_names()
        ds = CosemDataset(name)
    except urllib.error.URLError:
        # we don't want to fail the test suite just because the internet is down
        pytest.xfail("Internet down?")
        return
    except urllib.error.HTTPError as e:
        # could refine this error, but if 404, we want to be notified.
        raise AssertionError("Dataset not found") from e

    assert ds.name == name == str(ds)
    assert isinstance(ds.metadata, dict)
    assert isinstance(ds.description, str)
    assert isinstance(repr(ds), str)
    views = ds.views
    assert views
    assert isinstance(views, dict) and all(isinstance(v, dict) for v in views.values())
    first_view = list(views.values())[0]["name"]
    assert isinstance(ds.view(first_view), dict)
    view = ds.load_view(name=first_view)
    assert isinstance(view, xr.DataArray)

    images = ds.images
    assert images and isinstance(images, dict)

    data = ds.read_image(list(images)[0], level=2)
    assert isinstance(data, xr.DataArray)

