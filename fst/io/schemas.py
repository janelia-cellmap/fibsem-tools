import zarr
from typing import Union, Any, List
from typing_extensions import TypedDict
from fst.io import read
from pathlib import Path
import json


def jsonify(obj: Union[str, zarr.hierarchy.group, zarr.hierarchy.array]) -> dict:
    """
    Recursively serialize the structure + metadata of an n5/zarr group or array to json
    """

    class Node(TypedDict):
        attrs: str
        nodes: List

    val: Node = {"attrs": "", "nodes": []}

    if isinstance(obj, str):
        try:
            obj = read(obj)
        except ValueError:
            return dict(val)

    result = {}

    attrs_path = Path(obj.store.path) / Path(obj.path) / "attributes.json"
    attrs = json.loads(attrs_path.read_text())
    val["attrs"] = attrs
    key = obj.basename
    result[key] = val
    if hasattr(obj, "values"):
        result[key]["nodes"] = [jsonify(child) for child in obj.values()]

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Represent a zarr/n5 hierarchy as JSON"
    )
    parser.add_argument("path")
    args = parser.parse_args()
    print(jsonify(args.path))
