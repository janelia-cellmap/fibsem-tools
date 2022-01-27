from pydantic import BaseModel
from typing import List


class InstanceName(BaseModel):
    long: str
    short: str


class AnnotationState(BaseModel):
    present: bool
    annotated: bool


class Label(BaseModel):
    value: int
    name: InstanceName
    annotationState: AnnotationState


class LabelList(BaseModel):
    labels: List[Label]


classNameDict = {
    1: InstanceName(short="ECS", long="Extracellular Space"),
    2: InstanceName(short="Plasma membrane", long="Plasma membrane"),
    3: InstanceName(short="Mito membrane", long="Mitochondrial membrane"),
    4: InstanceName(
        short="Mito lumen",
        long="Mitochondrial lumen",
    ),
    5: InstanceName(short="Mito DNA", long="Mitochondrial DNA"),
    6: InstanceName(short="Golgi Membrane", long="Golgi apparatus membrane"),
    7: InstanceName(short="Golgi lumen", long="Golgi apparatus lumen"),
    8: InstanceName(short="Vesicle membrane", long="Vesicle membrane"),
    9: InstanceName(short="Vesicle lumen", long="VesicleLumen"),
    10: InstanceName(short="MVB membrane", long="Multivesicular body membrane"),
    11: InstanceName(short="MVB lumen", long="Multivesicular body lumen"),
    12: InstanceName(short="Lysosome membrane", long="Lysosome membrane"),
    13: InstanceName(short="Lysosome lumen", long="Lysosome membrane"),
    14: InstanceName(short="LD membrane", long="Lipid droplet membrane"),
    15: InstanceName(short="LD lumen", long="Lipid droplet lumen"),
    16: InstanceName(short="ER membrane", long="Endoplasmic reticulum membrane"),
    17: InstanceName(short="ER lumen", long="Endoplasmic reticulum membrane"),
    18: InstanceName(
        short="ERES membrane", long="Endoplasmic reticulum exit site membrane"
    ),
    19: InstanceName(short="ERES lumen", long="Endoplasmic reticulum exit site lumen"),
    20: InstanceName(short="NE membrane", long="Nuclear envelope membrane"),
    21: InstanceName(short="NE lumen", long="Nuclear envelope lumen"),
    22: InstanceName(short="Nuclear pore out", long="Nuclear pore out"),
    23: InstanceName(short="Nuclear pore in", long="Nuclear pore in"),
    24: InstanceName(short="HChrom", long="Heterochromatin"),
    25: InstanceName(short="NHChrom", long="Nuclear heterochromatin"),
    26: InstanceName(short="EChrom", long="Euchromatin"),
    27: InstanceName(short="NEChrom", long="Nuclear euchromatin"),
    28: InstanceName(short="Nucleoplasm", long="Nucleoplasm"),
    29: InstanceName(short="Nucleolus", long="Nucleolus"),
    30: InstanceName(short="Microtubules out", long="Microtubules out"),
    31: InstanceName(short="Centrosome", long="Centrosome"),
    32: InstanceName(short="Distal appendages", long="Distal appendages"),
    33: InstanceName(short="Subdistal appendages", long="Subdistal appendages"),
    34: InstanceName(short="Ribosomes", long="Ribsoomes"),
    35: InstanceName(short="Cytosol", long="Cytosol"),
    36: InstanceName(short="Microtubules in", long="Microtubules in"),
    37: InstanceName(short="Nucleus combined", long="Nucleus combined"),
    38: InstanceName(short="Actin", long="Actin"),
    39: InstanceName(short="T bar", long="T bar"),
    40: InstanceName(
        short="HChrom Point Annot.", long="Heterochromatin point annotation"
    ),
    41: InstanceName(short="EChrom Point Annot.", long="Euchromatin point annotation"),
}
