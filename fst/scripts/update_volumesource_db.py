from os import name

from typing import Sequence
from dataclasses import dataclass
import xarray
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Mapping
from fst.io.io import infer_dtype
from fst.io import read, DataArrayFromFile
from dataclasses import asdict
from fst.attrs import (
    VolumeSource,
    DisplaySettings,
    ContrastLimits,
    SpatialTransform,
    DatasetView,
    MeshSource,
)
from pymongo import MongoClient, ReplaceOne
from sheetscrape.scraper import GoogleSheetScraper
import pandas as pd
from fst.attrs import DatasetView
import zarr
import click

# for volume names, which underscore-separated trailing values are suffixes:
# e.g., 'mito_seg' has 'seg' as a suffix, while 'mito_er_contacts' has no suffix and is itself an entire token (for now)
volume_name_suffixes = ('seg', 'pred')

# constants for getting stuff from google docs
credfile = '/groups/scicompsoft/home/bennettd/certificates/cosem-db-1669c2970c7f.json'
view_sheet_name = 'FIB-SEM dataset regions of interest'
view_sheet_page = 0


def dir_glob(path: str) -> Tuple[str, ...]:
    return tuple(sorted(map(str, Path(path).glob("*"))))


def make_segmentation_display_settings(path: str, prediction_colors: Dict[str, str]):
    return DisplaySettings(
        contrastLimits=ContrastLimits(0, 1), color=prediction_colors[Path(path).name]
    )

# groundtruth is .5 x the grid spacing of raw data
ground_truth_scaling = 0.5
base_res = 4.0
yx_res = {"y": base_res, "x": base_res}
base_display = DisplaySettings(ContrastLimits(0, 1.0), "white", True)

axes = ("z", "y", "x")
units = ("nm",) * len(axes)
translate = (0,) * len(axes)


def scale(
    factors: Union[float, Sequence[float]], transform: SpatialTransform
) -> SpatialTransform:
    return SpatialTransform(
        transform.axes,
        transform.units,
        transform.translate,
        tuple(np.multiply(factors, transform.scale)),
    )

def infer_container_type(path: str) -> str:
    if Path(path).suffix == ".mrc":
        containerType = "mrc"
    elif any(map(lambda v: (Path(v).suffix == ".n5"), Path(path).parts)):
        containerType = "n5"
    elif any(map(lambda v: (Path(v).suffix == ".precomputed"), Path(path).parts)):
        containerType = "precomputed"
    else:
        raise ValueError(f"Could not infer container type from path {path}")

    return containerType

# todo: fix this generating the wrong info for analysis volumes
def get_classname_and_content_type(volume_name: str, suffixes: Sequence[str] = volume_name_suffixes) -> Tuple[str, str]:
    # remove all whitespace before splitting on _
    separated = ''.join(volume_name.split()).split("_")
    if separated[-1] in suffixes:
        class_name = separated[0]
    else:
        class_name = volume_name 
    content_type = ""
    if separated[-1] == "pred":
        content_type = "prediction"
    elif separated[-1] == "seg":
        content_type = "segmentation"
    else:
        content_type = "analysis"
    return class_name, content_type


base_transforms = {
    "jrc_hela-4": SpatialTransform(
        axes, units, translate, (base_res * 1.07, base_res, base_res)
    ),
    "jrc_mus-pancreas-1": SpatialTransform(
        axes, units, translate, (base_res * 0.85, base_res, base_res)
    ),
    "jrc_hela-2": SpatialTransform(
        axes, units, translate, (base_res * 1.31, base_res, base_res)
    ),
    "jrc_hela-3": SpatialTransform(
        axes, units, translate, (base_res * 0.81, base_res, base_res)
    ),
    "jrc_jurkat-1": SpatialTransform(
        axes, units, translate, (base_res * 0.86, base_res, base_res)
    ),
    "jrc_macrophage-2": SpatialTransform(
        axes, units, translate, (base_res * 0.84, base_res, base_res)
    ),
    "jrc_sum159-1": SpatialTransform(
        axes, units, translate, (base_res * 1.14, base_res, base_res)
    ),
    "jrc_ctl-id8-1": SpatialTransform(
        axes, units, translate, (base_res * 0.87, base_res, base_res)
    ),
    "jrc_fly-fsb-1": SpatialTransform(
        axes, units, translate, (base_res * 1.0, base_res, base_res)
    ),
    "jrc_fly-acc-calyx-1": SpatialTransform(
        axes, units, translate, (base_res * 0.93, base_res, base_res)
    ),
    "jrc_hela-1": SpatialTransform(axes, units, translate, (base_res * 2,) * 3),
    "jrc_choroid-plexus-2": SpatialTransform(
        axes, units, translate, (base_res * 2,) * 3
    ),
    "jrc_cos7-11": SpatialTransform(axes, units, translate, (base_res * 2,) * 3),
}

class_info = {
    "cent": ("Centrosome", "white"),
    "cent-dapp": ("Centrosome Distal Appendage", "white"),
    "cent-sdapp": ("Centrosome Subdistal Appendage", "white"),
    "chrom": ("Chromatin", "white"),
    "er": ("Endoplasmic Reticulum", "blue"),
    "er_palm" : ("Light Microscopy (PALM) of ER", "white"),
    "er_sim" : ("Light Microscopy (SIM) of the ER", "white"), 
    "er-mem": ("Endoplasmic Reticulum membrane", "blue"),
    "eres": ("Endoplasmic Reticulum Exit Site", "white"),
    "eres-mem": ("Endoplasmic Reticulum Exit Site membrane", "white"),
    "endo": ("Endosomal Network", "magenta"),
    "endo-mem": ("Endosome membrane", "magenta"),
    "echrom": ("Euchromatin", "white"),
    "ecs": ("Extracellular Space", "white"),
    "golgi": ("Golgi", "cyan"),
    "golgi-mem": ("Golgi membrane", "cyan"),
    "hchrom": ("Heterochromatin", "white"),
    "ld": ("Lipid Droplet", "white"),
    "ld-mem": ("Lipid Droplet membrane", "white"),
    "lyso": ("Lysosome", "white"),
    "lyso-mem": ("Lysosome membrane", "white"),
    "mt": ("Microtubule", "orange"),
    "mt-in": ("Microtubule inner", "orange"),
    "mt-out": ("Microtubule outer", "orange"),
    "mito": ("Mitochondria", "green"),
    "mito_palm" : ("Light Microscopy (PALM) of Mitochondria", "white"),
    "mito_sim" : ("Light Microscopy (SIM) of Mitochondria", "white"), 
    "mito-mem": ("Mitochondria membrane", "green"),
    "mito-ribo": ("Mitochondria Ribosome", "white"),
    "ne": ("Nuclear Envelope", "white"),
    "ne-mem": ("Nuclear Envelope membrane", "white"),
    "np": ("Nuclear Pore", "white"),
    "np-in": ("Nuclear Pore inner", "white"),
    "np-out": ("Nuclear Pore outer", "white"),
    "nucleolus": ("Nucleoulus", "white"),
    "nechrom": ("Nucleoulus associated Euchromatin", "white"),
    "nhchrom": ("Nucleoulus associated Heterochromatin", "white"),
    "nucleus": ("Nucleus", "red"),
    "pm": ("Plasma Membrane", "orange"),
    "ribo": ("Ribosome", "yellow"),
    "vesicle": ("Vesicle", "red"),
    "vesicle-mem": ("Vesicle membrane", "red"),
    "er_ribo_contacts": ("ER - Ribosome Contact Sites", "white"),
    "er_golgi_contacts": ("ER - Golgi Contact Sites", "white"),
    "er_mito_contacts": ("ER - Mito Contact Sites", "white"),
    "endo_er_contacts": ("Endosome - ER Contact Sites", "white"),
    "er_nucleus_contacts": ("ER - Nucleus Contact Sites", "white"),
    "er_pm_contacts": ("ER - Plasma Membrane Contat Sites", "white"),
    "er_vesicle_contacts": ("ER - Vesicle Contact Sites", "white"),
    "golgi_vesicle_contacts": ("Golgi - Vesicle Contact Sites", "white"),
    "endo_golgi_contacts": ("Endosome - Golgi Contact Sites", "white"),
    "mito_pm_contacts": ("Mito - Plasma Membrane Contact Sites", "white"),
    "er_mt_contacts": ("ER - Microtubule Contact Sites", "white"),
    "endo_mt_contacts": ("Endosome - Microtubule Contact Sites", "white"),
    "golgi_mt_contacts": ("Golgi - Microtubule Contact Sites", "white"),
    "mito_mt_contacts": ("Mitochondria - Microtubule Contact Sites", "white"),
    "mt_nucleus_contacts": ("Microtubule - Nucleus Contact Sites", "white"),
    "mt_vesicle_contacts": ("Microtubule - Vesicle Contact Sites", "white"),
    "mt_pm_contacts": ("Microtubule - Plasma Membrane Contact Sites", "white"),
    "mito_skeleton": ("Mitochrondria Skeletons", "white"),
    "mito_skeleton-lsp": ("Mitochrondria Skeletons: Longest Shortest Path", "white"),
    "er_medial-surface": ("ER Medial Surface", "white"),
    "er_curvature": ("Reconstructed ER from Medial Surface with Curvature", "white"),
    'ribo_classified' : ('Ribosomes classified by contact surface', 'white'),
    "fibsem-uint8": ("FIB-SEM Data (compressed)", "white"),
    "fibsem-uint16": ("FIB-SEM Data (uncompressed)", "white"),
    "gt": ("Ground truth", "white"),
}



def get_views_from_google_docs(credfile, spreadsheet_name, spreadsheet_page):
    sheet = GoogleSheetScraper(credfile, mode='r').client.open(spreadsheet_name)
    # convert the sheet to a dataframe
    columns, *values = sheet.worksheets()[spreadsheet_page].get_all_values()
    df = pd.DataFrame(values, columns=columns)
    return views_from_dataframe(df)

def views_from_dataframe(df: pd.DataFrame):
    views = []
    for idx,r in df.iterrows():
        datasetName: str = r['Dataset name']
        name: str = r['ROI Name (short)']
        description: str = r['ROI Description (long)']
        pos_str: str = r['XYZ Coordinates (nm)']
        pos: Optional[Sequence[int]]
        scale_str: str = r['Cross-Section Scale']
        scale: Optional[float]

        if len(pos_str) > 0:
            pos = tuple(map(int, ''.join(pos_str.split()).split(',')))
        else:
            pos = None
        
        if scale_str == '':
            scale = None
        else:
            scale = float(scale_str)

        layers: List[str] = r['Layers'].split(', ')
        for l in layers:
            class_name, _ = get_classname_and_content_type(l)
            if class_name not in class_info:
                print(f'Warning: layer name {l} from the view called {name} was not found in the list of layers.')
        views.append(DatasetView(datasetName, name, description, pos, scale, layers))
    return views


@dataclass
class RawSources:
    datasetName: str
    uint8: Optional[Tuple[str, DisplaySettings]] = None
    uint16: Optional[Tuple[str, DisplaySettings]] = None
    pred: Optional[Tuple[str, ...]] = None
    groundTruth: Optional[str] = None
    meshes: Tuple[str, ...] = ()


hess_raw_dir = "/groups/hess/hesslab/HighResPaper_rawdata"
cosem_raw_dir = "/groups/cosem/cosem/data"
raw_sources = (
    RawSources(
        "jrc_hela-4",
        (
            f"{hess_raw_dir}/10. HeLa_mitotic_17-7_Cell4_4x4x4nm/HeLa_mitotic_17-7_17_Cell4 4x4x4nm_raw.mrc",
            DisplaySettings(ContrastLimits(0.57, 0.805)),
        ),
        (
            f"{hess_raw_dir}/10. HeLa_mitotic_17-7_Cell4_4x4x4nm/HeLa_mitotic_17-7_17_Cell4 4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.275, 0.355), invertColormap=True),
        ),
    ),
    RawSources(
        "jrc_mus-pancreas-1",
        (
            f"{hess_raw_dir}/1. Pancreas Islet_G64-2-1-HighGlucose_4x4x4nm/G64-2-1_HighGlucose 4x4x4nm.mrc",
            DisplaySettings(ContrastLimits(0.7176, 0.8117)),
        ),
        (
            f"{hess_raw_dir}/1. Pancreas Islet_G64-2-1-HighGlucose_4x4x4nm/G64-2-1_HighGlucose 4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.108, 0.199), invertColormap=True),
        ),
    ),
    RawSources(
        "jrc_hela-2",
        (
            f"{cosem_raw_dir}/HeLa_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm.mrc",
            DisplaySettings(ContrastLimits(0.415, 0.716)),
        ),
        (
            f"{hess_raw_dir}/2. HeLa2_Aubrey_17-7_17_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.373, 0.52), invertColormap=True),
        ),
        pred=dir_glob(
            "/groups/cosem/cosem/ackermand/forDavis/jrc_hela-2/jrc_hela-2.n5"
        ),
        groundTruth="/nrs/cosem/bennettd/groundtruth/jrc_hela-2/jrc_hela-2.n5/groundtruth_0003/",
        meshes=dir_glob('/nrs/cosem/bennettd/s3/janelia-cosem/jrc_hela-2/neuroglancer/mesh')
    ),
    RawSources(
        "jrc_hela-3",
        (
            f"{hess_raw_dir}/3. HeLa3_Aubrey_17-7_17_Cell3_4x4x4nm/HeLa_Cell3_17-7_17_4x4x4nm.mrc",
            DisplaySettings(ContrastLimits(0.216, 0.944)),
        ),
        (
            f"{hess_raw_dir}/3. HeLa3_Aubrey_17-7_17_Cell3_4x4x4nm/HeLa_Cell3_17-7_17_4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.313, 0.409), invertColormap=True),
        ),
        pred=dir_glob(
            "/groups/cosem/cosem/ackermand/forDavis/jrc_hela-3/jrc_hela-3.n5"
        ),
        groundTruth="/nrs/cosem/bennettd/groundtruth/jrc_hela-3/jrc_hela-3.n5/groundtruth_0003/",
    ),
    RawSources(
        "jrc_jurkat-1",
        (
            f"{hess_raw_dir}/7. Jurkat_Cryo_2017_FS96_Cell1_4x4x4nm/Jurkat_Cryo_2017_FS96_Area1 4x4x4nm.mrc",
            DisplaySettings(ContrastLimits(0.826, 0.924)),
        ),
        (
            f"{hess_raw_dir}/7. Jurkat_Cryo_2017_FS96_Cell1_4x4x4nm/Jurkat_Cryo_2017_FS96_Area1 4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.297, 0.402), invertColormap=True),
        ),
        pred=dir_glob(
            "/groups/cosem/cosem/ackermand/forDavis/jrc_jurkat-1/jrc_jurkat-1.n5"
        ),
        groundTruth="/nrs/cosem/bennettd/groundtruth/jrc_jurkat-1/jrc_jurkat-1.n5/groundtruth_0003/",
    ),
    RawSources(
        "jrc_macrophage-2",
        (
            f"{cosem_raw_dir}/Macrophage_FS80_Cell2_4x4x4nm/Cryo_FS80_Cell2 4x4x4nm.mrc",
            DisplaySettings(ContrastLimits(0.843, 0.917)),
        ),
        (
            f"{hess_raw_dir}/6. Macrophage_FS80_Cell2_4x4x4nm/Macrophage_FS80_Cell2 4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.297, 0.436), invertColormap=True),
        ),
        pred=dir_glob(
            "/groups/cosem/cosem/ackermand/forDavis/jrc_macrophage-2/jrc_macrophage-2.n5"
        ),
        groundTruth="/nrs/cosem/bennettd/groundtruth/jrc_macrophage-2/jrc_macrophage-2.n5/groundtruth_0003/",
    ),
    RawSources(
        "jrc_sum159-1",
        (
            f"{hess_raw_dir}/8. SUM159_WT45_Cell2_4x4x4nm/SUM159_WT45_Cell2_Cryo_20171009_4x4x4nm.mrc",
            DisplaySettings(ContrastLimits(0.706, 0.864)),
        ),
        (
            f"{hess_raw_dir}/8. SUM159_WT45_Cell2_4x4x4nm/SUM159_WT45_Cell2_Cryo_20171009_4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.241, 0.334), invertColormap=True),            
        ),
        groundTruth="/nrs/cosem/bennettd/groundtruth/jrc_sum159-1/jrc_sum159-1.n5/groundtruth_0003/"
    ),
    RawSources(
        "jrc_ctl-id8-1",
        None,
        (
            f"{hess_raw_dir}/9. TCell_on_cancer_4x4x4nm/TCell_on_cancer_Atlas1_4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.166, 0.265), invertColormap=True),
        ),
    ),
    RawSources(
        "jrc_fly-fsb-1",
        None,
        (
            f"{hess_raw_dir}/4. Fly Fan Shaped Body[Column1-9]_Z0519-11_4x4x4nm/FB-Z0519-11 4x4x4nm 16bit.mrc",
            DisplaySettings(ContrastLimits(0.03433, 0.0509), invertColormap=True),
        ),
    ),
    RawSources(
        "jrc_fly-acc-calyx-1",
        None,
        (
            "/groups/cosem/cosem/bennettd/imports/jrc_fly-acc-calyx-1/jrc_fly-acc-calyx-1.n5/aligned_uint16",
            DisplaySettings(ContrastLimits(0.02499, 0.04994), invertColormap=True),
        ),
    ),
    RawSources(
        "jrc_hela-1",
        (
            f"{cosem_raw_dir}/HeLa_Cell1_8x8x8nm/Aubrey_17-7_17_Cell1_D05-10_8x8x8nm.mrc",
            DisplaySettings(ContrastLimits(.39, .56), invertColormap=True),
        ),
        pred=dir_glob(
            "/groups/cosem/cosem/ackermand/forDavis/jrc_hela-1/jrc_hela-1.n5"
        ),
    ),
    RawSources(
        "jrc_choroid-plexus-2",
        (
            "/nrs/cosem/bennettd/Choroid-Plexus_8x8x8nm/Choroid-Plexus_8x8x8nm.n5/volumes/raw/s0/",
            DisplaySettings(ContrastLimits(0, 1)),
        ),
        pred=dir_glob(
            "/groups/cosem/cosem/ackermand/forDavis/jrc_choroid-plexus-2/jrc_choroid-plexus-2.n5"
        ),
    ),
    RawSources(
        "jrc_cos7-11",
        None,
        (
            f"/nrs/cosem/bennettd/COS7_Cell11_8x8x8nm/SIFTalignTrans-invert.n5/volumes/raw/",
            DisplaySettings(ContrastLimits(0, 1)),
        ),
    ),
)

def makeVolumeSource(
    datasetName: str,
    volumeName: str,
    path: str,
    displaySettings: DisplaySettings,
    transform: SpatialTransform,
    contentType: str,
    description: str,
    version: str = "0",
    tags: Optional[Tuple[str]] = None,
) -> Optional[VolumeSource]:

    try: 
        arr: xarray.DataArray = DataArrayFromFile(path)
    except zarr.errors.PathNotFoundError as e:
        print(f'Could not access an array at {path}')
        return None

    dimensions: Tuple[int, ...] = arr.shape
    containerType = infer_container_type(path)
    return VolumeSource(
        path=path,
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


def process_raw_sources(
    raw_sources: Sequence[RawSources], class_info: Mapping[str, Tuple[str, str]]
) -> List[Union[VolumeSource, MeshSource]]:
    results: List[Union[VolumeSource, MeshSource, None]] = []
    for raw_source in raw_sources:
        datasetName = raw_source.datasetName
        base_transform = base_transforms[datasetName]

        uint8 = raw_source.uint8
        if uint8 is not None:
            volumeName = "fibsem-uint8"
            path, displaySettings = uint8
            description, _ = class_info[volumeName]
            results.append(
                makeVolumeSource(
                    datasetName=datasetName,
                    volumeName=volumeName,
                    path=path,
                    displaySettings=displaySettings,
                    transform=base_transform,
                    contentType="em",
                    description=description,
                )
            )

        uint16 = raw_source.uint16
        if uint16 is not None:
            volumeName = "fibsem-uint16"
            description, _ = class_info[volumeName]
            path, displaySettings = uint16
            results.append(
                makeVolumeSource(
                    datasetName=datasetName,
                    volumeName=volumeName,
                    path=path,
                    displaySettings=displaySettings,
                    transform=base_transform,
                    contentType="em",
                    description=description,
                )
            )

        pred = raw_source.pred
        if pred is not None:
            volume_paths = pred
            for vp in volume_paths:
                volumeName = Path(vp).name
                className, contentType = get_classname_and_content_type(volumeName)
                description, color = class_info[className]
                displaySettings = DisplaySettings(
                    contrastLimits=ContrastLimits(0, 1), color=color
                )
                
                # the 8nm datasets yield predictions that are 4nm grid spacing
                if base_transform.scale[0] == 8.0:
                    transform = scale(.5, base_transform)
                else:
                    transform = base_transform
                 
                results.append(
                    makeVolumeSource(
                        datasetName,
                        volumeName=volumeName,
                        path=vp,
                        displaySettings=displaySettings,
                        transform=transform,
                        contentType=contentType,
                        description=description,
                    )
                )
        gt = raw_source.groundTruth
        if gt is not None:
            path = gt
            volumeName = "gt"
            description, color = class_info[volumeName]
            displaySettings = DisplaySettings(ContrastLimits(0, 1), color=color)
            results.append(
                makeVolumeSource(
                    datasetName,
                    volumeName=volumeName,
                    path=gt,
                    displaySettings=displaySettings,
                    transform=scale(ground_truth_scaling, base_transform),
                    contentType="segmentation",
                    description=description,
                )
            )
        meshes = raw_source.meshes
        for mp in meshes:
            meshName = Path(mp).name
            results.append(MeshSource(path=mp, 
                                      name=meshName, 
                                      datasetName=datasetName,
                                      format='neuroglancer_precomputed_mesh'))
    
    return list(filter(lambda v: v is not None, results))


def upsert_sources_to_db(sources: Sequence[Union[VolumeSource, MeshSource]]):
    un = "root"
    pw = "root"
    db_name = "cosem"
    # use the datasetName/volumeName as the id for the document
    volumeSources = []
    meshSources = []
    for s in sources:
        f = asdict(s)
        if isinstance(s, VolumeSource):
            f.update({"_id": f["datasetName"] + "/" + f["name"]})
            volumeSources.append(f)
        elif isinstance(s, MeshSource):
            f.update({"_id": f["datasetName"] + "/" + f["name"]})
            meshSources.append(f)
        else:
            f"Object of type {type(f)} cannot be inserted into the database"

    # insert each element in the list into the `datasets` collection on our MongoDB instance
    with MongoClient(f"mongodb://{un}:{pw}@{db_name}.int.janelia.org") as client:        
        # this clearly sucks
        db = client["sources"]["VolumeSource"]
        operations = [
            ReplaceOne(filter={"_id": doc["_id"]}, replacement=doc, upsert=True)
            for doc in volumeSources
        ]
        db.bulk_write(operations)
        
        db = client["sources"]["MeshSource"]
        operations = [
            ReplaceOne(filter={"_id": doc["_id"]}, replacement=doc, upsert=True)
            for doc in meshSources
        ]
        db.bulk_write(operations)
        
        return True


def upsert_views_to_db(dataset_views: Sequence[DatasetView]):
    un = "root"
    pw = "root"
    db_name = "cosem"
    flat = list(map(asdict, dataset_views))
    # use the datasetName/volumeName as the id for the document
    [f.update({"_id": f["datasetName"] + "/" + f["name"]}) for f in flat]

    # insert each element in the list into the `datasets` collection on our MongoDB instance
    with MongoClient(f"mongodb://{un}:{pw}@{db_name}.int.janelia.org") as client:
        db = client["sources"]["DatasetView"]
        operations = [
            ReplaceOne(filter={"_id": doc["_id"]}, replacement=doc, upsert=True)
            for doc in flat
        ]
        result = db.bulk_write(operations)
        return result

@click.command()
@click.option("-v", "--views", required=False, type=bool)
@click.option("-s", "--sources", required=False, type=bool)
def db_update_cli(views: bool, sources: bool):
    result = None
    if sources:        
        volume_sources = process_raw_sources(raw_sources, class_info)
        print(f'Updating database with {len(volume_sources)} sources')
        result = upsert_sources_to_db(volume_sources)
    
    if views:
        dataset_views = get_views_from_google_docs(credfile, view_sheet_name, view_sheet_page)
        print(f'Updating view database with {len(dataset_views)} views')
        result = upsert_views_to_db((dataset_views))        
    
    return result
if __name__ == "__main__":
    db_update_cli()    
