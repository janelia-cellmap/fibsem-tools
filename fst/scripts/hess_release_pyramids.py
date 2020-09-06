from operator import mul
from xarray import DataArray
import xarray
from fst.io.mrc import mrc_to_dask
from fst.attrs import display_attrs
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, List
from fst.io import read
from mrcfile.mrcmemmap import MrcMemmap
import dask.array as da

# groundtruth is .5 x the grid spacing of raw data
ground_truth_scaling = .5
base_res = 4.0
yx_res = {"y": base_res, "x": base_res}
base_display = display_attrs(0, 1.0, 1.0, "white", True)
FORMATS = {'n5','precomputed'}
CONTENT_TYPES = {'em','lm','prediction','segmentation'}
voxel_sizes = {'jrc_hela-4' : {"z": base_res * 1.07, **yx_res},
            'jrc_mus-pancreas-1': {"z": base_res * 0.85, **yx_res},
            'jrc_hela-2': {"z": base_res * 1.31, **yx_res},
            'jrc_hela-3': {"z": base_res * 0.81, **yx_res},
            'jrc_jurkat-1' : {"z": base_res * 0.86, **yx_res},
            "jrc_macrophage-2": {"z": base_res * 0.84, **yx_res},
            "jrc_sum159-1": {"z": base_res * 1.14, **yx_res},
            "jrc_ctl-id8-1": {"z": base_res * 0.87, **yx_res},
            'jrc_fly-fsb-1': {"z": base_res * 1.0, **yx_res},
            'jrc_fly-acc-calyx-1': {"z": base_res * 0.93, **yx_res}}

prediction_containers = {'jrc_hela-2': '/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/HeLa2.n5',
                        'jrc_hela-3': '/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/HeLa3.n5',
                        'jrc_jurkat-1': '/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/Jurkat.n5',
                        'jrc_macrophage-2': '/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/Macrophage.n5'
                        }

prediction_colors = {'er': 'blue',
                     'golgi': 'cyan',
                     'mito': 'green',
                     'mvb': 'magenta',
                     'plasma_membrane': 'orange',
                     'vesicle' : 'red',
                    }

prediction_descriptions = {'er': 'Endoplasmic reticulum segmentation',
                          'golgi': 'Golgi segmentation',
                          'mito': "Mitochondria segmentation",
                          'mvb' : 'Multivesicular body segmentation',
                          'plasma_membrane': 'Plasma membrane segmentation',
                          'vesicle' : 'Vesicle segmentation',
                          }

def subsample_reduce(a, axis=None):
    """
    Coarsening by subsampling, compatible with da.coarsen
    """
    if axis is None:
        return a
    else:
        samples = []
        for ind,s in enumerate(a.shape):
            if ind in axis:
                samples.append(slice(0, 1, None))
            else:
                samples.append(slice(None))        
        return a[tuple(samples)].squeeze()

def mode_reduce(a, axis=None):
    """
    Coarsening by computing the n-dimensional mode, compatible with da.coarsen
    """
    from scipy.stats import mode
    if axis is None:
         return a
    else:
        transposed = a.transpose(*range(0, a.ndim, 2), *range(1, a.ndim, 2))
        reshaped = transposed.reshape(*transposed.shape[:a.ndim//2], -1)
        modes = mode(reshaped, axis=reshaped.ndim-1).mode
        return modes.squeeze()


@dataclass 
class StorageSpec:
    driver: str
    store_format: str
    outer_path: str
    inner_path: str

    def toURI(self):
        return f'{self.driver}://{Path(self.outer_path).with_suffix("." + self.store_format).joinpath(self.inner_path)}'

    def __post_init__(self):
        if self.store_format not in FORMATS:
            raise ValueError(f'store_format must be one of {FORMATS}')

@dataclass
class OpenOrganelleDataset:
    name: str
    volumes: Dict[str, str]

@dataclass
class MultiscalePlan:
    reduction: Callable
    depth: int
    compute: bool

@dataclass
class Source:
    datasetName: str
    description: str
    contentType: str
    sourcePath: str
    destPath: str
    destFormat: str    
    resolution: dict
    displaySettings: dict
    multiscalePlan: MultiscalePlan
    roi: list
    tags: list
    alignment: str = '0'
    mutation: Optional[Callable] = None

    def toDataArray(self):
        if Path(self.sourcePath).suffix == ".mrc":
            array = mrc_to_dask(self.sourcePath, chunks=(1, -1, -1))
        else:
            r = read(self.sourcePath)
            array = da.from_array(r, chunks=r.chunks)
        coords = [
            DataArray(
                np.arange(array.shape[idx]) * self.resolution[k],
                dims=k,
                attrs={"units": "nm"},
            )
            for idx, k in enumerate(self.resolution)
        ]
        return DataArray(
            array, coords=coords)

def contract(arr: xarray.DataArray, output_shape: dict):
    return arr.isel({k: slice(0, output_shape[k]) for k in output_shape.keys()})

def flip_axis(arr: xarray.DataArray, axis: str):
    arr2 = arr.copy()
    idx = arr2.dims.index(axis)
    arr2.data = np.flip(arr2.data, idx)
    return arr2

raw_sources = [
        Source(
            datasetName = "jrc_hela-4",
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/10. HeLa_mitotic_17-7_Cell4_4x4x4nm/HeLa_mitotic_17-7_17_Cell4 4x4x4nm_raw.mrc",
            description="Raw EM Data",
            contentType='em',
            destPath = 'fibsem/aligned_uint8_v0',
            destFormat='precomputed',
            resolution=voxel_sizes["jrc_hela-4"],
            displaySettings=display_attrs(0.57, 0.805, invertColormap=False),
            multiscalePlan=MultiscalePlan(np.mean, 5, False),
            roi = [],
            tags = [],
        ),

    Source(datasetName = 'jrc_mus-pancreas-1',
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/1. Pancreas Islet_G64-2-1-HighGlucose_4x4x4nm/G64-2-1_HighGlucose 4x4x4nm.mrc",
            description="Raw EM Data",
            contentType='em',
            destPath="fibsem/aligned_uint8_v0",
            destFormat='precomputed',
            resolution=voxel_sizes['jrc_mus-pancreas-1'],
            displaySettings=display_attrs(0.1, 0.22, invertColormap=True),
            multiscalePlan=MultiscalePlan(np.mean, 5, False),
            roi = [],
            tags = [],
        ),
    
    Source(datasetName="jrc_hela-2",
            sourcePath="/groups/cosem/cosem/data/HeLa_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm.mrc",
            description="Raw EM Data",
            contentType='em',
            destPath="fibsem/aligned_uint8_v0",
            destFormat='precomputed',
            resolution=voxel_sizes['jrc_hela-2'],
            displaySettings=display_attrs(0.415, 0.716, invertColormap=False),
            multiscalePlan=MultiscalePlan(np.mean, 5, False),
            roi = [],
            tags = [],
        ),
        Source(datasetName="jrc_hela-2",
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/2. HeLa2_Aubrey_17-7_17_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm 16bit.mrc",
            description="Raw EM Data",
            contentType='em',
            destPath="fibsem/aligned_uint16_v0",
            destFormat='n5',
            resolution=voxel_sizes['jrc_hela-2'],
            displaySettings=display_attrs(0.415, 0.716, invertColormap=True),
            multiscalePlan=MultiscalePlan(np.mean, 5, False),
            roi = [],
            tags = [],
        ),
        
        Source(datasetName="jrc_hela-3",
            sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/3. HeLa3_Aubrey_17-7_17_Cell3_4x4x4nm/HeLa_Cell3_17-7_17_4x4x4nm.mrc",
            description="Raw EM Data",
            contentType='em',
            destPath="fibsem/aligned_uint8_v0",
            destFormat='precomputed',
            resolution=voxel_sizes['jrc_hela-3'],
            displaySettings=display_attrs(0.216, 0.944, invertColormap=False),
            multiscalePlan=MultiscalePlan(np.mean, 5, False),
            roi = [],
            tags = []
        ),

    Source(datasetName="jrc_jurkat-1",
        sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/7. Jurkat_Cryo_2017_FS96_Cell1_4x4x4nm/Jurkat_Cryo_2017_FS96_Area1 4x4x4nm.mrc",
        description="Raw EM Data",
        contentType='em',
        destPath="fibsem/aligned_uint8_v0",
        destFormat='precomputed',
        resolution=voxel_sizes['jrc_jurkat-1'],
        displaySettings=display_attrs(0.826, 0.924, invertColormap=False),
        multiscalePlan=MultiscalePlan(np.mean, 5, False),
        roi = [],
        tags = [],
    ),
    Source(datasetName="jrc_macrophage-2",
        sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/6. Macrophage_FS80_Cell2_4x4x4nm/Macrophage_FS80_Cell2 4x4x4nm.mrc",
        description="Raw EM Data",
        contentType='em',
        destPath="fibsem/aligned_uint8_v0",
        destFormat='precomputed',
        resolution=voxel_sizes['jrc_macrophage-2'],
        displaySettings=display_attrs(0.201, 0.744, invertColormap=False),
        multiscalePlan=MultiscalePlan(np.mean, 5, False),
        roi = [],
        tags = [],
    ),
    Source(datasetName="jrc_sum159-1",
        sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/8. SUM159_WT45_Cell2_4x4x4nm/SUM159_WT45_Cell2_Cryo_20171009_4x4x4nm.mrc",
        description="Raw EM Data",
        contentType='em',
        destPath="fibsem/aligned_uint8_v0",
        destFormat='precomputed',
        resolution=voxel_sizes['jrc_sum159-1'],
        displaySettings=display_attrs(0.706, 0.864, invertColormap=False),
        multiscalePlan=MultiscalePlan(np.mean, 5, False),
        roi = [],
        tags = [],
    ),
    Source(datasetName="jrc_ctl-id8-1",
        sourcePath="/groups/hess/hesslab/HighResPaper_rawdata/9. TCell_on_cancer_4x4x4nm/TCell_on_cancer_Atlas1_4x4x4nm.mrc",
        description="Raw EM Data",
        contentType='em',
        destPath="fibsem/aligned_uint8_v0",
        destFormat='precomputed',
        resolution=voxel_sizes["jrc_ctl-id8-1"],
        displaySettings=display_attrs(0.861, 0.949, invertColormap=False),
        multiscalePlan=MultiscalePlan(np.mean, 5, False),
        roi = [],
        tags = [],
    ),
    Source(datasetName="jrc_fly-fsb-1",
        sourcePath="/groups/cosem/cosem/bennettd/imports/jrc_fly-fsb-1/jrc_fly-fsb-1.n5/aligned_uint8",
        description="Raw EM Data",
        contentType='em',
        destPath="fibsem/aligned_uint8_v0",
        destFormat='precomputed',
        resolution=voxel_sizes['jrc_fly-fsb-1'],
        displaySettings=display_attrs(0.027, 0.069, invertColormap=True),
        multiscalePlan=MultiscalePlan(np.mean, 5, False),
        roi = [],
        tags = [],
    ),
    Source(datasetName="jrc_fly-acc-calyx-1",
        sourcePath="/groups/cosem/cosem/bennettd/imports/jrc_fly-acc-calyx-1/jrc_fly-acc-calyx-1.n5/aligned_uint16",
        description="Raw EM Data",
        contentType='em',
        destPath="fibsem/aligned_uint16_v0",
        destFormat='n5',
        resolution=voxel_sizes['jrc_fly-acc-calyx-1'],
        displaySettings=display_attrs(0.023, 0.051, invertColormap=True),
        multiscalePlan=MultiscalePlan(np.mean, 5, False),
        roi = [],
        tags = [],
    ),
    Source(datasetName="jrc_fly-acc-calyx-1",
        sourcePath="/groups/cosem/cosem/bennettd/imports/jrc_fly-acc-calyx-1/jrc_fly-acc-calyx-1.n5/aligned_uint8",
        description="Raw EM Data",
        contentType='em',
        destPath="fibsem/aligned_uint8_v0",
        destFormat='precomputed',
        resolution=voxel_sizes['jrc_fly-acc-calyx-1'],
        displaySettings=display_attrs(0.023, 0.051, invertColormap=True),
        multiscalePlan=MultiscalePlan(np.mean, 5, True),
        roi = [],
        tags = [],
    ),
    ]

prediction_sources: List[Source] = []
for name in prediction_containers: 
    for cls in prediction_descriptions:
        src = Source(datasetName=name,
                sourcePath=f"{prediction_containers[name]}/{cls}",
                description=prediction_descriptions[cls],
                contentType='segmentation',
                destPath=f"predictions/{cls}_seg",
                destFormat='n5',
                resolution=voxel_sizes[name],
                displaySettings=display_attrs(color=prediction_colors[cls]),
                multiscalePlan=MultiscalePlan(mode_reduce, 5, False),
                roi = [],
                tags = [],
            )
        
        if name == 'jrc_hela-2':
            src.mutation = lambda v: flip_axis(v, 'y')
        
        prediction_sources.append(src)
    
sources = prediction_sources + raw_sources
