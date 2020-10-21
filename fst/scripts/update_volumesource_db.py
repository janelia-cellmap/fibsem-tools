from typing import Sequence
from dataclasses import dataclass
import xarray
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Mapping
from fst.io.io import infer_dtype
from fst.io import read, DataArrayFromFile
from dataclasses import asdict
from fst.attrs import VolumeSource, DisplaySettings, ContrastLimits, SpatialTransform, DatasetView
from pymongo import MongoClient, ReplaceOne


def pred_glob(path: str) -> Tuple[str,...]:
    return tuple(sorted(map(str, Path(path).glob('*'))))


def make_segmentation_display_settings(path: str, prediction_colors: Dict[str, str]):
    return DisplaySettings(contrastLimits=ContrastLimits(0,1), color=prediction_colors[Path(path).name])


def make_description(path: str):

    return ''


# groundtruth is .5 x the grid spacing of raw data
ground_truth_scaling = 0.5
base_res = 4.0
yx_res = {"y": base_res, "x": base_res}
base_display = DisplaySettings(ContrastLimits(0, 1.0), "white", True)

axes = ('z','y','x')
units = ('nm',) * len(axes)
translate = (0,) * len(axes)

def scale(factors: Union[float, Sequence[float]], transform: SpatialTransform) -> SpatialTransform:
    return SpatialTransform(transform.axes, transform.units, transform.translate, tuple(np.multiply(factors, transform.scale)))


base_transforms = {
    "jrc_hela-4": SpatialTransform(axes, units, translate, (base_res * 1.07, base_res, base_res)),
    "jrc_mus-pancreas-1": SpatialTransform(axes, units, translate, (base_res * 0.85, base_res, base_res)),
    "jrc_hela-2": SpatialTransform(axes, units, translate, (base_res * 1.31, base_res, base_res)),
    "jrc_hela-3": SpatialTransform(axes, units, translate, (base_res * 0.81, base_res, base_res)),
    "jrc_jurkat-1": SpatialTransform(axes, units, translate, (base_res * 0.86, base_res, base_res)),
    "jrc_macrophage-2": SpatialTransform(axes, units, translate, (base_res * 0.84, base_res, base_res)),
    "jrc_sum159-1": SpatialTransform(axes, units, translate, (base_res * 1.14, base_res, base_res)),
    "jrc_ctl-id8-1": SpatialTransform(axes, units, translate, (base_res * 0.87, base_res, base_res)),
    "jrc_fly-fsb-1": SpatialTransform(axes, units, translate, (base_res * 1.0, base_res, base_res)),
    "jrc_fly-acc-calyx-1": SpatialTransform(axes, units, translate, (base_res * 0.93, base_res, base_res)),
    "jrc_hela-1": SpatialTransform(axes, units, translate, (base_res * 2,) * 3),
    "jrc_choroid-plexus-2": SpatialTransform(axes, units, translate, (base_res * 2,) * 3),
    "jrc_cos7-11": SpatialTransform(axes, units, translate, (base_res * 2,) * 3)
}

class_info = {
    'cent': ('Centrosome', 'white'),
    'cent-dapp' : ("Centrosome Distal Appendage", 'white'),
    'cent-sdapp' : ("Centrosome Subdistal Appendage", 'white'),
    'chrom' : ("Chromatin", 'white'),
    'er' : ("Endoplasmic Reticulum", 'blue'),
    'er-mem' : ("Endoplasmic Reticulum membrane", 'blue'),
    'eres' : ("Endoplasmic Reticulum Exit Site", 'white'),
    'eres-mem': ("Endoplasmic Reticulum Exit Site membrane", 'white'),
    'endo' : ("Endosomal Network", 'magenta'),
    'endo-mem' : ("Endosome membrane",'magenta'),
    'echrom' : ("Euchromatin",'white'),
    'ecs' : ("Extracellular Space",'white'),
    'golgi' : ("Golgi",'cyan'),
    'golgi-mem' : ("Golgi membrane",'cyan'),
    'hchrom' : ("Heterochromatin",'white'),
    'ld' : ("Lipid Droplet",'white'),
    'ld-mem' : ("Lipid Droplet membrane",'white'),
    'lyso' : ("Lysosome",'white'),
    'lyso-mem' : ("Lysosome membrane",'white'),
    'mt' : ("Microtubule",'orange'),
    'mt-in' : ("Microtubule inner",'orange'),
    'mt-out' : ("Microtubule outer",'orange'),
    'mito' : ("Mitochondria",'green'),
    'mito-mem' : ("Mitochondria membrane",'green'),
    'mito-ribo' : ("Mitochondria Ribosome",'white'),
    'ne' : ("Nuclear Envelope",'white'),
    'ne-mem' : ("Nuclear Envelope membrane",'white'),
    'np' : ("Nuclear Pore",'white'),
    'np-in' : ("Nuclear Pore inner",'white'),
    'np-out' : ("Nuclear Pore outer",'white'),
    'nucleolus' : ("Nucleoulus",'white'),
    'nechrom' : ("Nucleoulus associated Euchromatin",'white'),
    'nhchrom' : ("Nucleoulus associated Heterochromatin",'white'),
    'nucleus' : ("Nucleus",'red'),
    'pm' : ("Plasma Membrane",'orange'),
    'ribo' : ("Ribosome",'yellow'),
    'vesicle' : ("Vesicle",'red'),
    'vesicle-mem' : ("Vesicle membrane",'red'),
    'er_ribo_contacts' : ("ER - Ribosome Contact Sites",'white'),
    'er_golgi_contacts' : ("ER - Golgi Contact Sites",'white'),
    'er_mito_contacts' : ("ER - Mito Contact Sites",'white'),
    'endo_er_contacts' : ("Endosome - ER Contact Sites",'white'),
    'er_nucleus_contacts' : ("ER - Nucleus Contact Sites",'white'),
    'er_pm_contacts': ("ER - Plasma Membrane Contat Sites",'white'),
    'er_vesicle_contacts' : ("ER - Vesicle Contact Sites",'white'),
    'golgi_vesicle_contacts' : ("Golgi - Vesicle Contact Sites",'white'),
    'endo_golgi_contacts' : ("Endosome - Golgi Contact Sites",'white'),
    'mito_pm_contacts' : ("Mito - Plasma Membrane Contact Sites",'white'),
    'er_mt_contacts' : ("ER - Microtubule Contact Sites",'white'),
    'endo_mt_contacts' : ("Endosome - Microtubule Contact Sites",'white'),
    'golgi_mt_contacts' : ("Golgi - Microtubule Contact Sites",'white'),
    'mito_mt_contacts' : ("Mitochondria - Microtubule Contact Sites",'white'),
    'mt_nucleus_contacts' : ("Microtubule - Nucleus Contact Sites",'white'),
    'mt_vesicle_contacts' : ("Microtubule - Vesicle Contact Sites",'white'),
    'mt_pm_contacts': ("Microtubule - Plasma Membrane Contact Sites",'white'),
    'mito_skeleton' : ("Mitochrondria Skeletons",'white'),
    'er_medialsurface' : ("ER Medial Surface",'white'),
    'er_curvature': ("Reconstructed ER from Medial Surface with Curvature",'white'),
    "em-uint8": ("FIB-SEM Data (compressed)",'white'),
    "em-uint16": ("FIB-SEM Data (uncompressed)",'white'),
    "gt": ('Ground truth','white')
}

@dataclass
class RawSources:
    datasetName: str
    uint8: Optional[Tuple[str, DisplaySettings]] = None
    uint16: Optional[Tuple[str, DisplaySettings]] = None
    pred: Optional[Tuple[str,...]] = None
    groundTruth: Optional[str] = None
        
hess_raw_dir = "/groups/hess/hesslab/HighResPaper_rawdata"
cosem_raw_dir = '/groups/cosem/cosem/data'
raw_sources = (
    RawSources('jrc_hela-4',
               (f"{hess_raw_dir}/10. HeLa_mitotic_17-7_Cell4_4x4x4nm/HeLa_mitotic_17-7_17_Cell4 4x4x4nm_raw.mrc", 
                DisplaySettings(ContrastLimits(0.57, 0.805))),
               (f'{hess_raw_dir}/10. HeLa_mitotic_17-7_Cell4_4x4x4nm/HeLa_mitotic_17-7_17_Cell4 4x4x4nm 16bit.mrc',
              DisplaySettings(ContrastLimits(0.275, 0.355), invertColormap=True))),
    RawSources('jrc_mus-pancreas-1',
               (f"{hess_raw_dir}/1. Pancreas Islet_G64-2-1-HighGlucose_4x4x4nm/G64-2-1_HighGlucose 4x4x4nm.mrc",
               DisplaySettings(ContrastLimits(0.68, 0.82))),
               (f"{hess_raw_dir}/1. Pancreas Islet_G64-2-1-HighGlucose_4x4x4nm/G64-2-1_HighGlucose 4x4x4nm 16bit.mrc", 
               DisplaySettings(ContrastLimits(.108, 0.169), invertColormap=True))),
    RawSources('jrc_hela-2', 
               (f"{cosem_raw_dir}/HeLa_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm.mrc", 
               DisplaySettings(ContrastLimits(0.415, 0.716))),
               (f"{hess_raw_dir}/2. HeLa2_Aubrey_17-7_17_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm 16bit.mrc",
               DisplaySettings(ContrastLimits(0.373, 0.52), invertColormap=True)),
               pred=pred_glob("/groups/cosem/cosem/ackermand/forDavis/jrc_hela-2/jrc_hela-2.n5"),
               groundTruth="/nrs/cosem/davis/groundtruth/jrc_hela-2/jrc_hela-2.n5/groundtruth_0003/"),
    RawSources('jrc_hela-3',
              (f"{hess_raw_dir}/3. HeLa3_Aubrey_17-7_17_Cell3_4x4x4nm/HeLa_Cell3_17-7_17_4x4x4nm.mrc",
              DisplaySettings(ContrastLimits(0.216, 0.944))),
              (f'{hess_raw_dir}/3. HeLa3_Aubrey_17-7_17_Cell3_4x4x4nm/HeLa_Cell3_17-7_17_4x4x4nm 16bit.mrc', 
               DisplaySettings(ContrastLimits(0.297, .402), invertColormap=True)),
              pred=pred_glob("/groups/cosem/cosem/ackermand/forDavis/jrc_hela-3/jrc_hela-3.n5"),
              groundTruth="/nrs/cosem/davis/groundtruth/jrc_hela-3/jrc_hela-3.n5/groundtruth_0003/"),
    RawSources('jrc_jurkat-1',
              (f"{hess_raw_dir}/7. Jurkat_Cryo_2017_FS96_Cell1_4x4x4nm/Jurkat_Cryo_2017_FS96_Area1 4x4x4nm.mrc",
              DisplaySettings(ContrastLimits(0.826, 0.924))),
              (f"{hess_raw_dir}/7. Jurkat_Cryo_2017_FS96_Cell1_4x4x4nm/Jurkat_Cryo_2017_FS96_Area1 4x4x4nm 16bit.mrc",
              DisplaySettings(ContrastLimits(0.297, .402), invertColormap=True)),
              pred=pred_glob("/groups/cosem/cosem/ackermand/forDavis/jrc_jurkat-1/jrc_jurkat-1.n5"),
              groundTruth="/nrs/cosem/davis/groundtruth/jrc_jurkat-1/jrc_jurkat-1.n5/groundtruth_0003/"),
    RawSources('jrc_macrophage-2',
              (f"{hess_raw_dir}/6. Macrophage_FS80_Cell2_4x4x4nm/Macrophage_FS80_Cell2 4x4x4nm.mrc", 
               DisplaySettings(ContrastLimits(0.201, 0.744))),
              (f"{hess_raw_dir}/6. Macrophage_FS80_Cell2_4x4x4nm/Macrophage_FS80_Cell2 4x4x4nm 16bit.mrc", 
              DisplaySettings(ContrastLimits(0.297, .436), invertColormap=True)),
              pred=pred_glob("/groups/cosem/cosem/ackermand/forDavis/jrc_macrophage-2/jrc_macrophage-2.n5"),
              groundTruth="/nrs/cosem/davis/groundtruth/jrc_macrophage-2/jrc_macrophage-2.n5/groundtruth_0003/"),
    RawSources('jrc_sum159-1',
              (f"{hess_raw_dir}/8. SUM159_WT45_Cell2_4x4x4nm/SUM159_WT45_Cell2_Cryo_20171009_4x4x4nm.mrc", 
              DisplaySettings(ContrastLimits(0.706, 0.864))),
              (f"{hess_raw_dir}/8. SUM159_WT45_Cell2_4x4x4nm/SUM159_WT45_Cell2_Cryo_20171009_4x4x4nm 16bit.mrc",
              DisplaySettings(ContrastLimits(.241, .334), invertColormap=True))),
    RawSources('jrc_ctl-id8-1',
              (f"{hess_raw_dir}/9. TCell_on_cancer_4x4x4nm/TCell_on_cancer_Atlas1_4x4x4nm.mrc", 
              DisplaySettings(ContrastLimits(0.861, 0.949))),
              (f"{hess_raw_dir}/9. TCell_on_cancer_4x4x4nm/TCell_on_cancer_Atlas1_4x4x4nm 16bit.mrc",
              DisplaySettings(ContrastLimits(.166, .265), invertColormap=True))),
    RawSources('jrc_fly-fsb-1',
              ("/groups/cosem/cosem/bennettd/imports/jrc_fly-fsb-1/jrc_fly-fsb-1.n5/aligned_uint8",
              DisplaySettings(ContrastLimits(0.634, 0.816))),
              (f"{hess_raw_dir}/4. Fly Fan Shaped Body[Column1-9]_Z0519-11_4x4x4nm/FB-Z0519-11 4x4x4nm 16bit.mrc", 
               DisplaySettings(ContrastLimits(.0310, .062), invertColormap=True))),
    RawSources('jrc_fly-acc-calyx-1',
              None,
              ("/groups/cosem/cosem/bennettd/imports/jrc_fly-acc-calyx-1/jrc_fly-acc-calyx-1.n5/aligned_uint16", 
              DisplaySettings(ContrastLimits(0.023, 0.051), invertColormap=True))),
    RawSources('jrc_hela-1',
              (f"{cosem_raw_dir}/HeLa_Cell1_8x8x8nm/Aubrey_17-7_17_Cell1_D05-10_8x8x8nm.mrc", 
               DisplaySettings(ContrastLimits(0, 1))),
               pred=pred_glob("/groups/cosem/cosem/ackermand/forDavis/jrc_hela-1/jrc_hela-1.n5")),
    RawSources('jrc_choroid-plexus-2',
              (f"{cosem_raw_dir}/Choroid-Plexus_4x4x4nm/8x8x8nm/Walsh#1 8x8x8nm.mrc", 
               DisplaySettings(ContrastLimits(0, 1))),
               pred=pred_glob("/groups/cosem/cosem/ackermand/forDavis/jrc_choroid-plexus-2/jrc_choroid-plexus-2.n5")),
    RawSources('jrc_cos7-11',
              (f"{cosem_raw_dir}/COS7_Cell11_8x8x8nm/Cryo_LoadID277_Cell11_8x8x8nm_bigwarped_v17.n5/volumes/raw/", 
               DisplaySettings(ContrastLimits(0, 1))),),
              )

defaultView = DatasetView('Default view', 'An overview of data volumes.',  None, None, [])

roi_views = {'jrc_hela-3': 
        [DatasetView('Centrosome', 'Centrosome of a wild-type HeLa cell, as described in Fig. 2b of Xu et al., 2021.', [28620, 3187, 10706], 10, volumeKeys=['fibsem-uint8']),
        DatasetView('Golgi stack', 'Golgi Stack of a wild-type HeLa cell, as described in Fig. 2c of Xu et al., 2021.', [30800, 2553, 11522], 10, volumeKeys=['fibsem-uint8']),
        DatasetView('Nuclear envelope with polyribosomes', '', [24948, 800, 11083], 10, volumeKeys=['fibsem-uint8'])
        ],
        'jrc_ctl-id8-1' : [
            DatasetView('Cytotoxic T Lymphocytes cupping a cancel cell', 
                        'A cytotoxic T Lymphocyte cupping the target ovarian cancer cell, as described in Fig. 3b of Xu et al., 2021.', 
                        [41798, 4354, 20829],
                        10, 
                        ['fibsem-uint8']),
            DatasetView('Centrosome',	
                        'The centrosome of a cytotoxic T Lymphocyte polarized toward the target ovarian cell, as described in Fig. 3c of Xu et al. 2021.', [37911, 7445, 19380],
                         10,
                         ['fibsem-uint8']),
            DatasetView('Lytic granule', 
                        'The detailed structure of a lytic granule in a killer immune T cell, as described in Fig. 3d of Xu et al., 2021.', 
                        [36975, 7159, 14744], 
                        10, 
                        ['fibsem-uint8']),
            DatasetView('Membrane interdigitation', 
                        'Membrane interdigitation of the interface between a cytotoxic T Lymphocyte and the target ovarian cancer cell, as described in Fig. S16a of Xu et al., 2021.', [37955, 3708, 20211], 10, ['fibsem-uint8']),
            DatasetView('Flat membrane apposition', 'Flat membrane apposition of the interface between a cytotoxic T Lymphocyte and the target ovarian cancer cell, as described in Fig. S16b of Xu et al., 2021.', [39530, 3483, 21345], 10, ['fibsem-uint8']),
            DatasetView('Filopodia of a cancer cell', 'Filopodia of the target ovarian cancer cell trapped between a cytotoxic T Lymphocyte and the target cell, as described in Fig. S16c of Xu et al., 2021.', [40872, 4768, 20921], 10, ['fibsem-uint8']),
        ],
        'jrc_mus-pancreas-1': [
            DatasetView('Primary cilium', 'A primary cilium with axoneme and centrioles of wild-type mouse pancreatic islet cells treated with 16.7 mM glucose, as described in Fig. 4b of Xu et al., 2021.', [8098, 8107, 109], 3.3087, volumeKeys=['fibsem-uint8']),
            DatasetView('Microvilli', 'Intermingled microvilli of wild-type mouse pancreatic islet cells treated with 16.7 mM glucose, as described in Fig. 4c of Xu et al., 2021.', [10461, 16355, 21887], 3.3254, volumeKeys=['fibsem-uint8']),
            DatasetView('Secretory granules', 'Ultrastructural diversity among insulin secretory granules containing rod-shaped or spherical crystals of wild-type mouse pancreatic islet cells treated with 16.7 mM glucose, as described in Fig. 4d of Xu et al., 2021.', [20118, 12950, 20750], 3.3254, volumeKeys=['fibsem-uint8']),
            DatasetView('Secretory granules and endoplasmic reticulum', 'Close contacts between secretory granules and endoplasmic reticulum of wild-type mouse pancreatic islet cells treated with 16.7 mM glucose, as described in Fig. 4e of Xu et al., 2021.', [21020, 5399, 19329], 3.0601, volumeKeys=['fibsem-uint8'])
        ],
        'jrc_fly-acc-calyx-1' : [
            DatasetView('Presynaptic T-bar in Accessory Calyx', 'Presynaptic T-bar in accessory calyx of a five-day-old adult Drosophila brain', [11802, 25455, 53993], 5.749, volumeKeys=['fibsem-uint16']),
            DatasetView('Output sites of a projection neuron', 'Output sites of a projection neuron in accessory calyx of a five-day-old adult Drosophilabrain', [29119, 27630, 46442], 10, volumeKeys=['fibsem-uint16'])
        ],
        'jrc_fly-fsb-1' : [
         DatasetView('Dense-core vesicles (DCVs)', 'Dense-core vesicles (DCVs) of different sizes in fan-shaped body of a five-day-old adult Drosophilabrain, as described in Fig. 4h of Xu et al., 2021.', [14336, 26200, 20640], 1.875, volumeKeys=['fibsem-uint8']),
         DatasetView('Multiple synaptic sites (T-bars)', 'Multiple synaptic sites (T-bars) viewed from the side in fan-shaped body of a five-day-old adult Drosophilabrain, as described in Fig. 4i of Xu et al., 2021.', [7120, 42124, 20800], 1.875, volumeKeys=['fibsem-uint8']),
         DatasetView('Presynaptic sites (T-bars)', 'A presynaptic T-bar viewed from the top in fan-shaped body of a five-day-old adult Drosophilabrain, as described in Fig. 4j of Xu et al., 2021.', [23880, 33640, 20816], 1.875, volumeKeys=['fibsem-uint8']),
         DatasetView('Microtubules', 'A longitudinal-section view of microtubules in fan-shaped body of a five-day-old adult Drosophilabrain, as described in Fig. 4k of Xu et al., 2021.', [4660, 26716, 30828], 1.875, volumeKeys=['fibsem-uint8']),
         DatasetView('Microtubule arrays', 'A cross-section view of microtubule arrays in fan-shaped body of a five-day-old adult Drosophilabrain, as described in Fig. 4l of Xu et al., 2021.', [20720, 30200, 36572], 1.875, volumeKeys=['fibsem-uint8'])   
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
    
    views[k].append(defaultView)
    if k in roi_views:
        views[k].extend(roi_views[k])
 


def makeVolumeSource(datasetName: str,
                    volumeName: str, 
                    path: str, 
                    displaySettings: DisplaySettings, 
                    transform: SpatialTransform, 
                    contentType: str, 
                    description: str, 
                    version: str='0', 
                    tags: Optional[Tuple[str]]= None) -> VolumeSource:

    arr: xarray.DataArray = DataArrayFromFile(path)
    dimensions: Tuple[int, ...] = arr.shape    
    containerType = infer_container_type(path)
    return VolumeSource(path=path,
                    datasetName=datasetName,
                    name=volumeName,
                    dataType=str(arr.dtype),
                    dimensions=dimensions,
                    displaySettings=displaySettings, 
                    transform=transform, 
                    contentType=contentType,
                    containerType=containerType,
                    description=description,
                    version=version,
                    tags=tags)

def infer_container_type(path: str) -> str:
    if Path(path).suffix=='.mrc':
        containerType='mrc'
    elif any(map(lambda v: (Path(v).suffix == '.n5'), Path(path).parts)):
        containerType='n5'
    else:
        raise ValueError(f'Could not infer container type from path {path}')

    return containerType

def get_classname_and_content_type(volume_name: str) -> Tuple[str,str]:
    separated = volume_name.split('_')
    classname = separated[0]
    content_type = ''
    if separated[-1]  == 'pred':
        content_type = 'prediction'
    elif separated[-1] == 'seg':
        content_type = 'segmentation'
    else:
        content_type = 'analysis'
    return classname, content_type


def process_raw_sources(raw_sources: Sequence[RawSources], class_info: Mapping[str, Tuple[str, str]]) -> List[VolumeSource]:
    results = []
    for raw_source in raw_sources:
        datasetName = raw_source.datasetName
        base_transform = base_transforms[datasetName]
        
        uint8 = raw_source.uint8
        if uint8 is not None:
            volumeName = 'fibsem-uint8'
            path, displaySettings = uint8
            description, _ = class_info['em-uint8'] 
            results.append(makeVolumeSource(datasetName=datasetName,
                                            volumeName=volumeName, 
                                            path=path, 
                                            displaySettings=displaySettings,
                                            transform=base_transform,
                                            contentType='em',
                                            description=description))
        
        uint16 = raw_source.uint16
        if uint16 is not None:
            volumeName = 'fibsem-uint16'
            description, _ = class_info['em-uint16']
            path, displaySettings = uint16
            results.append(makeVolumeSource(datasetName=datasetName,
                                            volumeName=volumeName, 
                                            path=path, 
                                            displaySettings=displaySettings,
                                            transform=base_transform,
                                            contentType='em',
                                            description=description))
        
        pred = raw_source.pred
        if pred is not None:
            volume_paths = pred
            for vp in volume_paths:
                volumeName = Path(vp).name
                
                className, contentType = get_classname_and_content_type(volumeName)
                description, color = class_info[className]
                displaySettings = DisplaySettings(contrastLimits=ContrastLimits(0,1), color=color)
                results.append(makeVolumeSource(datasetName,
                                               volumeName=volumeName,
                                               path=vp,
                                               displaySettings=displaySettings,
                                               transform=base_transform,
                                               contentType=contentType,
                                               description=description))
        gt = raw_source.groundTruth
        if gt is not None:
            path = gt
            volumeName = 'gt'
            description, color = class_info[volumeName]
            displaySettings=DisplaySettings(ContrastLimits(0,1), color=color)
            results.append(makeVolumeSource(datasetName,
                                           volumeName=volumeName,
                                           path=gt,
                                           displaySettings=displaySettings,
                                           transform=scale(ground_truth_scaling, base_transform),
                                           contentType='segmentation', 
                                           description=description))
    return results


def upsert_sources_to_db(volume_sources: Sequence[VolumeSource]):
    un = 'root'
    pw = 'root'
    db_name = 'cosem'
    flat = list(map(asdict, volume_sources))
    # use the volumeName as the id for the document
    [f.update({'_id': f['datasetName'] + '/' + f['name']}) for f in flat]

    # insert each element in the list into the `datasets` collection on our MongoDB instance
    with MongoClient(f'mongodb://{un}:{pw}@{db_name}.int.janelia.org') as client:
        db = client['VolumeSources']['to_ingest']
        operations = [ReplaceOne(filter={"_id": doc["_id"]}, replacement=doc, upsert=True) for doc in flat]
        result = db.bulk_write(operations)
        return result




if __name__ == '__main__':
    volume_sources = process_raw_sources(raw_sources, class_info)
    result = upsert_sources_to_db(volume_sources)
    print(result.acknowledged)