from pathlib import Path
from typing import List, Optional
import numpy as np
from xarray import DataArray


def display_attrs(
    contrast_min: float = 0,
    contrast_max: float = 1,
    gamma: float = 1,
    color: str = "white",
    invertColormap: bool = False
):
    assert (contrast_min >= 0) & (contrast_min <= 1)
    assert (contrast_max >= 0) & (contrast_max <= 1)
    assert gamma > 0
    return {
        "displaySettings": {
            "contrastMin": contrast_min,
            "contrastMax": contrast_max,
            "gamma": gamma,
            "color": color,
            "invertColormap": invertColormap,
        }
    }


def neuroglancer_multiscale_group_attrs(axes: List, units: List) -> dict:
    # see https://github.com/google/neuroglancer/issues/176#issuecomment-553027775
    return {"axes": axes, "units": units}


def cosem_array_attrs(
    translate: List, scale: List, units: List, axes: List, name: str
) -> dict:
    return {
        "name": name,
        "transform": {
            "translate": translate,
            "scale": scale,
            "units": units,
            "axes": axes,
        },
    }


def cosem_multiscale_group_attrs(
    transforms: List, array_paths: List, name: str
) -> dict:
    return {
        "name": name,
        "multiscales": [
            {
                "datasets": [
                    {"path": p, "transform": t} for p, t in zip(array_paths, transforms)
                ]
            }
        ],
    }


def n5v_array_attrs(dimensions: List, unit: str) -> dict:
    return {"pixelResolution": {"dimensions": dimensions, "unit": unit}}


def n5v_multiscale_group_attrs(scales: List[List], pixelResolution: dict) -> dict:
    return {"scales": scales, **pixelResolution}


def group_attrs(pyramids: List, array_paths: Optional[List] = None, axis_order='C') -> dict:
    if not array_paths:
        array_paths = [f"s{idx}" for idx in range(len(pyramids))]
    assert len(pyramids) == len(array_paths)

    if axis_order == 'F':
        axes = pyramids[0].dims[::-1]
        scales = [list(s.scale_factors)[::-1] for s in pyramids]        
    else:
        axes = pyramids[0].dims
        scales = [list(s.scale_factors) for s in pyramids]        
    coords_reordered = [pyramids[0].coords[k] for k in axes]
    units = list(d.units for d in coords_reordered)
    arr_attrs = [array_attrs(p, axis_order=axis_order) for p in pyramids]
    
    # we need this for n5-view... I think
    pixelResolution = {"pixelResolution": arr_attrs[0]["pixelResolution"]}
    # these settings determine downstream visualization of the data
    displaySettings = {"displaySettings": arr_attrs[0]["displaySettings"]}
    n5v_attrs = n5v_multiscale_group_attrs(scales, pixelResolution=pixelResolution)
    neuroglancer_attrs = neuroglancer_multiscale_group_attrs(
        axes=axes, units=units
    )
    transforms = [a["transform"] for a in arr_attrs]
    cosem_attrs = cosem_multiscale_group_attrs(
        transforms=transforms, array_paths=array_paths, name=arr_attrs[0]["name"]
    )
    return {**n5v_attrs, **neuroglancer_attrs, **cosem_attrs, **displaySettings}


def array_attrs(arr: DataArray, axis_order='C') -> dict:    
    if axis_order == 'F':
        axes = arr.dims[::-1]
    else:
        axes = arr.dims
    coords_reordered = [arr.coords[k] for k in axes]
    translate = [float(c.data[0]) for c in coords_reordered]
    scale = list(
        map(
            float,
            np.subtract([k.data[1] for k in coords_reordered], translate).tolist(),
        )
    )
    units = list(d.units for d in coords_reordered)
    name = arr.attrs["name"]
    display = arr.attrs.get("displaySettings")
    if not display:
        display = display_attrs()
        print("displaySettings not found in dataset attrs. Using defaults!")
    else:
        display = {'displaySettings': display}
        
    cosem_attrs = cosem_array_attrs(
        translate=translate, scale=scale, units=units, axes=axes, name=name
    )
    n5v_attrs = n5v_array_attrs(dimensions=scale, unit=units[0])
    return {**cosem_attrs, **n5v_attrs, **display}
