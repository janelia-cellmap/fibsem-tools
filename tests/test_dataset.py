from fibsem_tools._dataset import CosemDataset
import urllib.error
import pytest
import xarray as xr


def test_dataset():
    try:
        name = CosemDataset.all_names()[0]
        ds = CosemDataset(name)
    except urllib.error.URLError:
        # we don't want to fail the test suite just because the internet is down
        pytest.xfail("Internet down?")
        return
    except urllib.error.HTTPError as e:
        # could refine this error, but if 404, we want to be notified.
        raise AssertionError("Dataset not found") from e

    assert ds.name == name == str(ds)
    assert isinstance(ds.manifest, dict)
    assert isinstance(ds.title, str)
    assert isinstance(repr(ds), str)
    assert isinstance(ds.metadata, dict)
    views = ds.views
    assert views
    assert isinstance(views, list) and all(isinstance(v, dict) for v in views)
    first_view = views[0]["name"]
    assert isinstance(ds.view(first_view), dict)
    view = ds.load_view(name=first_view)
    assert isinstance(view, xr.DataArray)

    sources = ds.sources
    assert sources and isinstance(sources, dict)

    data = ds.read_source(list(sources)[0], level=1)
    assert isinstance(data, xr.DataArray)
