# from fibsem_tools.metadata.groundtruth import GroupMetadata
from fibsem_tools import read_xarray
import json
from fibsem_tools.metadata.groundtruth import (
    AnnotationEncoding,
    classNameDict,
    AnnotationArrayAttrs,
    AnnotationClassAttrs,
    AnnotationCropAttrs,
)
from rich import print_json
import numpy as np
import datetime
from typing import Dict, TypedDict, List, TypeVar

Key = TypeVar("Key", bound=str)


class CropMeta(TypedDict):
    maxId: int
    name: str
    offset: List[float]
    offset_unit: str
    resolution: List[float]
    resulution_unit: str
    type: str


dataset = "jrc_hela-2"
bucket = "janelia-cosem-datasets"
uri = f"s3://{bucket}/{dataset}/{dataset}.n5/labels/gt/"
out_dtype = "uint8"
out_dtype_max = np.iinfo(out_dtype).max

crop_key: Key = "Crop13"
group = read_xarray(uri)
arr = group["s0"].data
subvolumeMeta: Dict[Key, CropMeta] = arr.attrs["subvolumes"]
sMeta = subvolumeMeta[crop_key]
dims = ("x", "y", "z")

scales = arr.attrs["transform"]["scale"][::-1]
offsets = np.multiply(sMeta["offset"], np.divide(scales, sMeta["resolution"]))
selecter = {
    d: (np.arange(100) * scale) + offset
    for d, offset, scale in zip(dims, offsets, scales)
}

crop = arr.sel(selecter, method="nearest")
crop_attrs = AnnotationCropAttrs(
    name=crop_key, description="A crop", protocol=None, doi=None
)

out_attrs = {}
out_attrs[f"/{crop_key}"] = {"annotation": crop_attrs.dict()}
# partition the subvolume into separate integer classes
vals = np.unique(crop)


for v in vals:
    name, description = classNameDict[v].short, classNameDict[v].long

    subvol = (crop == v).astype(out_dtype)
    census = {k: np.sum(subvol == k) for k in np.unique(subvol)}
    encoding: AnnotationEncoding = {"absent": 0, "unknown": 255}
    array_attrs = AnnotationArrayAttrs(census=census, encoding=encoding, object=name)

    group_attrs = AnnotationClassAttrs(
        name=name,
        description=description,
        created_by=[
            "Cellmap annotators",
        ],
        created_with=["Amira", "Paintera"],
        start_date=datetime.datetime.now().isoformat(),
        duration_days=10,
        encoding=encoding,
        type="instance",
    )

    out_attrs[f"/{crop_key}/{name}"] = {"annotation": group_attrs.dict()}
    out_attrs[f"/{crop_key}/{name}/s0"] = {"annotation": array_attrs.dict()}


print_json(json.dumps(out_attrs))
