from numpy.lib.function_base import disp
from fst.attrs.attrs import DatasetView
from operator import mul
from xarray import DataArray
import xarray
from fst.io.mrc import mrc_to_dask, mrc_shape_dtype_inference
from fst.attrs import display_attrs
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple, Union
from fst.io import read
from mrcfile.mrcmemmap import MrcMemmap
import dask.array as da
from fst.pyramid import mode_reduce
from typing import Literal

from fst.attrs import VolumeIngest, VolumeSource

def flip_axis(arr: xarray.DataArray, axis: str):
    arr2 = arr.copy()
    idx = arr2.dims.index(axis)
    arr2.data = np.flip(arr2.data, idx)
    return arr2


# groundtruth is .5 x the grid spacing of raw data
ground_truth_scaling = 0.5
base_res = 4.0
yx_res = {"y": base_res, "x": base_res}
base_display = display_attrs(0, 1.0, 1.0, "white", True)

voxel_sizes = {
    "jrc_hela-4": {"z": base_res * 1.07, **yx_res},
    "jrc_mus-pancreas-1": {"z": base_res * 0.85, **yx_res},
    "jrc_hela-2": {"z": base_res * 1.31, **yx_res},
    "jrc_hela-3": {"z": base_res * 0.81, **yx_res},
    "jrc_jurkat-1": {"z": base_res * 0.86, **yx_res},
    "jrc_macrophage-2": {"z": base_res * 0.84, **yx_res},
    "jrc_sum159-1": {"z": base_res * 1.14, **yx_res},
    "jrc_ctl-id8-1": {"z": base_res * 0.87, **yx_res},
    "jrc_fly-fsb-1": {"z": base_res * 1.0, **yx_res},
    "jrc_fly-acc-calyx-1": {"z": base_res * 0.93, **yx_res},
    "jrc_hela-1": {"z": base_res * 2, "y": base_res * 2, "x": base_res * 2},
    "jrc_choroid-plexus-2": {"z": base_res * 2, "y": base_res * 2, "x": base_res * 2},
    "jrc_cos7-11": {"z": base_res * 2, "y": base_res * 2, "x": base_res * 2}
}

     

# '/groups/hess/hesslab/HighResPaper_rawdata/4. Fly Fan Shaped Body[Column1-9]_Z0519-11_4x4x4nm/FB-Z0519-11 4x4x4nm 16bit.mrc'
# '/groups/hess/hesslab/HighResPaper_rawdata/5. Fly_AccessoryCalyx_Z0519-15_4x4x4nm/AceeCalyx_Z0519-15 4x4x4nm 16bit.mrc.lnk'


raw_source_paths: Dict[str, Tuple[VolumeSource, ...]] = {
    "jrc_hela-4": (
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/10. HeLa_mitotic_17-7_Cell4_4x4x4nm/HeLa_mitotic_17-7_17_Cell4 4x4x4nm_raw.mrc",
            displaySettings=display_attrs(0.57, 0.805, invertColormap=False)),

        VolumeSource(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/10. HeLa_mitotic_17-7_Cell4_4x4x4nm/HeLa_mitotic_17-7_17_Cell4 4x4x4nm 16bit.mrc',
        displaySettings=display_attrs(0.275, 0.355, invertColormap=True)),
    ),
    "jrc_mus-pancreas-1": (
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/1. Pancreas Islet_G64-2-1-HighGlucose_4x4x4nm/G64-2-1_HighGlucose 4x4x4nm.mrc",
            displaySettings=display_attrs(0.68, 0.82, invertColormap=False)),

            VolumeSource(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/1. Pancreas Islet_G64-2-1-HighGlucose_4x4x4nm/G64-2-1_HighGlucose 4x4x4nm 16bit.mrc',
                     displaySettings=display_attrs(.108, 0.169, invertColormap=True))
    ),
    "jrc_hela-2": (
        VolumeSource(
            sourcePath="/groups/cosem/cosem/data/HeLa_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm.mrc",
            displaySettings=display_attrs(0.415, 0.716, invertColormap=False),
        ),
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/2. HeLa2_Aubrey_17-7_17_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm 16bit.mrc",
            displaySettings=display_attrs(0.373, 0.52, invertColormap=True),
        ),
    ),
    "jrc_hela-3": (
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/3. HeLa3_Aubrey_17-7_17_Cell3_4x4x4nm/HeLa_Cell3_17-7_17_4x4x4nm.mrc",
            displaySettings=display_attrs(0.216, 0.944, invertColormap=False),
        ),

        VolumeSource(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/3. HeLa3_Aubrey_17-7_17_Cell3_4x4x4nm/HeLa_Cell3_17-7_17_4x4x4nm 16bit.mrc',
        displaySettings=display_attrs(0.297, .402, invertColormap=True)
        )
    ),
    "jrc_jurkat-1": (
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/7. Jurkat_Cryo_2017_FS96_Cell1_4x4x4nm/Jurkat_Cryo_2017_FS96_Area1 4x4x4nm.mrc",
            displaySettings=display_attrs(0.826, 0.924, invertColormap=False),
        ),

        VolumeSource(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/7. Jurkat_Cryo_2017_FS96_Cell1_4x4x4nm/Cryo_2017_FS96_Area1 4x4x4nm 16bit.mrc',
        displaySettings=display_attrs(0.297, .402, invertColormap=True))
    ),
    "jrc_macrophage-2": (
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/6. Macrophage_FS80_Cell2_4x4x4nm/Macrophage_FS80_Cell2 4x4x4nm.mrc",
            displaySettings=display_attrs(0.201, 0.744, invertColormap=False)),

        VolumeSource(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/6. Macrophage_FS80_Cell2_4x4x4nm/Macrophage_FS80_Cell2 4x4x4nm 16bit.mrc',
        displaySettings=display_attrs(0.297, .436, invertColormap=True))
    ),
    "jrc_sum159-1": (
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/8. SUM159_WT45_Cell2_4x4x4nm/SUM159_WT45_Cell2_Cryo_20171009_4x4x4nm.mrc",
            displaySettings=display_attrs(0.706, 0.864, invertColormap=False),
        ),
        VolumeSource(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/8. SUM159_WT45_Cell2_4x4x4nm/SUM159_WT45_Cell2_Cryo_20171009_4x4x4nm 16bit.mrc',
        displaySettings=display_attrs(.241, .334, invertColormap=True))
    ),
    "jrc_ctl-id8-1": (
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/9. TCell_on_cancer_4x4x4nm/TCell_on_cancer_Atlas1_4x4x4nm.mrc",
            displaySettings=display_attrs(0.861, 0.949, invertColormap=False),
        ),
        VolumeSource(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/9. TCell_on_cancer_4x4x4nm/TCell_on_cancer_Atlas1_4x4x4nm 16bit.mrc',
        displaySettings=display_attrs(.166, .265, invertColormap=True))
    ),
    "jrc_fly-fsb-1": (
        VolumeSource(
            sourcePath="/groups/cosem/cosem/bennettd/imports/jrc_fly-fsb-1/jrc_fly-fsb-1.n5/aligned_uint8",
            displaySettings=display_attrs(0.634, 0.816, invertColormap=False),
        ),
        VolumeSource(
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/4. Fly Fan Shaped Body[Column1-9]_Z0519-11_4x4x4nm/FB-Z0519-11 4x4x4nm 16bit.mrc",
            displaySettings=display_attrs(.0310, .062, invertColormap=False),
        ),
    ),
    "jrc_fly-acc-calyx-1": (
        VolumeSource(
            sourcePath="/groups/cosem/cosem/bennettd/imports/jrc_fly-acc-calyx-1/jrc_fly-acc-calyx-1.n5/aligned_uint16",
            displaySettings=display_attrs(0.023, 0.051, invertColormap=True),
        ),

    ),
    "jrc_hela-1": (
        VolumeSource(sourcePath="/groups/cosem/cosem/data/HeLa_Cell1_8x8x8nm/'Aubrey_17-7_17_Cell1_D05-10_8x8x8nm.mrc",
        displaySettings=display_attrs()),
    ),
    
    "jrc_choroid-plexus-2" : (VolumeSource(sourcePath= "/groups/cosem/cosem/data/Choroid-Plexus_4x4x4nm/8x8x8nm/Walsh#1 8x8x8nm.mrc",
    displaySettings=display_attrs()),
    ),
    "jrc_cos7-11": (VolumeSource(sourcePath="/groups/cosem/cosem/data/COS7_Cell11_8x8x8nm/Cryo_LoadID277_Cell11_8x8x8nm_bigwarped_v17.n5/volumes/raw/", displaySettings=display_attrs()),) 
}



# This dataset is probably too large to save as a precomputed store without sharding, but the sharding API in tensorstore sucks right now
# VolumeSource(
#            sourcePath="/groups/cosem/cosem/bennettd/imports/jrc_fly-acc-calyx-1/jrc_fly-acc-calyx-1.n5/aligned_uint8",
#            displaySettings=display_attrs(0.023, 0.051, invertColormap=False),
#        ),

prediction_containers = {
    "jrc_hela-2": "/groups/cosem/cosem/ackermand/forDavis/HeLa2.n5",
    "jrc_hela-3": "/groups/cosem/cosem/ackermand/forDavis/HeLa3.n5",
    "jrc_jurkat-1": "/groups/cosem/cosem/ackermand/forDavis/Jurkat.n5",
    "jrc_macrophage-2": "/groups/cosem/cosem/ackermand/forDavis/Macrophage.n5",
}

prediction_colors = {
    "er": "blue",
    "golgi": "cyan",
    "mito": "green",
    "mvb": "magenta",
    "plasma_membrane": "orange",
    "vesicle": "red",
}

volume_descriptions = {
    "er": "ER",
    "golgi": "Golgi",
    "mito": "Mitochondria",
    "mvb": "Endosomal network",
    "plasma_membrane": "Plasma membrane",
    "vesicle": "Vesicles",
    "em_uint8": "FIB-SEM Data (compressed)",
    "em_uint16": "FIB-SEM Data (uncompressed)",
}

raw_volume_sources = []

for name in voxel_sizes.keys():
    raws = raw_source_paths[name]
    for rs in raws:
        if Path(rs.sourcePath).exists():
            dtype = infer_dtype(rs.sourcePath)
            destFormat=DTYPE_FORMATS[dtype]
            if destFormat == 'n5':
                destPath=f"{name}.n5/fibsem/aligned_{dtype}_v{rs.alignment}"
            elif destFormat == 'precomputed':
                destPath=f"neuroglancer/fibsem/aligned_{dtype}_v{rs.alignment}.{destFormat}"
            else:
                raise ValueError(f'Destination format {destFormat} could not be used to generate a path for data')
            src = VolumeIngest(
                datasetName=name,
                sourcePath=rs.sourcePath,
                description=volume_descriptions[f"em_{dtype}"],
                destPath=destPath,
                destFormat=destFormat,
                resolution=voxel_sizes[name],
                displaySettings=rs.displaySettings,
                multiscaleSpec=MultiscaleSpec(reduction='mean', depth=5, factors=2),
                alignment=rs.alignment,
                contentType="em",
                tags=[],
            )
            raw_volume_sources.append(src)

prediction_volume_sources: List[VolumeIngest] = []
for name in prediction_containers:
    for cls in prediction_colors:
        src = VolumeIngest(
            datasetName=name,
            sourcePath=f"{prediction_containers[name]}/{cls}",
            description=volume_descriptions[cls],
            contentType="segmentation",
            destPath=f"{name}.n5/predictions/{cls}_seg",
            destFormat="n5",
            resolution=voxel_sizes[name],
            displaySettings=display_attrs(color=prediction_colors[cls]),
            multiscaleSpec=MultiscaleSpec('mode', 5, 2),
            tags=[],
        )
        # these datasets were flipped along the y axis between mrc and n5 files, so all predictions are also flipped.
        # We will apply this flip to the predictions because neuroglancer puts position (0,0) in the top left corner of the image
        if name in ("jrc_hela-2", "jrc_hela-3", "jrc_macrophage-2", "jrc_jurkat-1"):
            src.mutation = "flip_y"

        prediction_volume_sources.append(src)

sources = prediction_volume_sources + raw_volume_sources
# sort by dataset name
sources = list(sorted(sources, key=lambda v: v.datasetName))

roi_views = {'jrc_hela-3': 
        [DatasetView('Centrosome', '', [28620, 3187, 10706], 10, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
        DatasetView('Golgi stack', '', [30800, 2553, 11522], 10, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
        DatasetView('Nuclear envelope with polyribosomes', '', [24948, 800, 11083], 10, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed'])
        ],
        'jrc_mus-pancreas-1': 
        [DatasetView('Primary cilium', '', [8098, 8107, 109], 3.3087, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
        DatasetView('Microvilli', '', [10461, 16355, 21887], 3.3254, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
        DatasetView('SG types', '', [20118, 12950, 20750], 3.3254, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
        DatasetView('SG + ER', '', [21020, 5399, 19329], 3.0601, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed'])
        ],
        'jrc_fly-acc-calyx-1' : [
            DatasetView('Presynaptic T-bar in Accessory Calyx', '', [11802, 25455, 53993], 5.749, volumeKeys=['jrc_fly-acc-calyx-1.n5/fibsem/aligned_uint16_v0']),
            DatasetView('Output sites of a projection neuron (PN) in Accessory Calyx', '', [29119, 27630, 46442], 10, volumeKeys=['jrc_fly-acc-calyx-1.n5/fibsem/aligned_uint16_v0'])
        ],
        'jrc_fly-fsb-1' : [
         DatasetView('Fig. 4h, Dense-Core Vesicles (DCVs) of different sizes', '', [14336, 26200, 20640], 1.875, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
         DatasetView('Fig. 4i, Multiple synaptic sites (T-bars) viewed from the side', '', [7120, 42124, 20800], 1.875, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
         DatasetView('Fig. 4j: A presynaptic T-bar viewed from the top', '', [23880, 33640, 20816], 1.875, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
         DatasetView('Fig. 4k: A longitudinal-section view of microtubules', '', [4660, 26716, 30828], 1.875, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed']),
         DatasetView('Fig. 4l: A cross-section view of microtubule arrays:', '', [20720, 30200, 36572], 1.875, volumeKeys=['neuroglancer/fibsem/aligned_uint8_v0.precomputed'])   
        ]
        }


# create views based on the sources
from itertools import groupby
raw_grouped, seg_grouped = {k: list(v) for k, v in groupby(raw_volume_sources, key=lambda v: v.datasetName)}, {k: list(v) for k, v in groupby(prediction_volume_sources, key=lambda v: v.datasetName)}
views: dict = {k: [] for k in raw_grouped}
for k in raw_grouped:
    # put in the default view    
    default_raw = raw_grouped[k][0]
    if k in seg_grouped:
        default_seg = seg_grouped[k]
    else: 
        default_seg = []
    defaultView = DatasetView('Default view', 'An overview of data volumes.',  None, None, [v.destPath for v in [default_raw, *default_seg]])
    views[k].append(defaultView)
    if k in roi_views:
        views[k].extend(roi_views[k])
    

