from typing import Iterator, Mapping, Tuple
from supabase import create_client
from datatree import DataTree
from fibsem_tools import read, read_xarray
from fibsem_tools.io.util import JSON
import os


def multiscale_datatree(group_url: str) -> DataTree:
    group = read(group_url)
    array_keys = sorted(
        tuple(name for name, _ in group.arrays()), key=lambda v: int(v[1:])
    )
    arrays = {
        arr_name: read_xarray(os.path.join(group_url, arr_name))
        for arr_name in array_keys
    }
    return DataTree.from_dict(arrays, name=group.basename)


def dataset_query_result_to_datatree(query_result) -> DataTree:
    data_dict = {
        img["name"]: multiscale_datatree(img["url"]) for img in query_result["image"]
    }
    return DataTree.from_dict(data_dict)


class Dataset(Mapping[str, DataTree]):
    supabase_url = "https://kvwjcggnkipomjlbjykf.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt2d2pjZ2dua2lwb21qbGJqeWtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NjUxODgyMjksImV4cCI6MTk4MDc2NDIyOX0.o_yLKX9erKbIrG3mwdwFkWYI8N9EjTNUnu9FWMngw9E"  # noqa: E501

    def __init__(self):
        self.client = create_client(self.supabase_url, self.supabase_key)

    def __keys__(self) -> Tuple[str]:
        query_result = self.client.table("dataset").select("name").execute()
        keys = [d["name"] for d in query_result.data]
        return keys

    def __getitem__(self, key: str) -> JSON:
        result = (
            self.client.table("dataset")
            .select("*, image(*)", count=1)
            .eq("name", key)
            .execute()
        )
        return result.data

    def __iter__(self) -> Iterator[str]:
        return (k for k in self.__keys__())

    def __len__(self) -> int:
        return 5


def test_dataset():
    d = Dataset()
    tuple(d.keys())
    good_result = d["jrc_mus-liver"]
    # this should keyerror
    d["foo"]
    # this is insanely slow lmao
    print(dataset_query_result_to_datatree(good_result[0]))


if __name__ == "__main__":
    test_dataset()
