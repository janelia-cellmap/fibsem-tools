from xarray import DataArray
from fst.io.mrc import mrc_to_dask
from fst.attrs import display_attrs
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Optional
from fst.io import read
from mrcfile.mrcmemmap import MrcMemmap
import dask.array as da

base_res = 4.0
yx_res = {'y': base_res, 'x': base_res}
base_display = display_attrs(0, 1.0, 1.0, 'white', True)

@dataclass
class Source:
    content: str
    sourcePath: str
    destPath: str    
    resolution: dict
    displaySettings: dict
    computePyramid: bool
    ancestor: Optional[str] = None
    mutation: Optional[str] = None 
    
    def toDataArray(self):        
        if Path(self.sourcePath).suffix == '.mrc':
            array = mrc_to_dask(self.sourcePath, chunks=(1,-1,-1))
        else:
            r = read(self.sourcePath)            
            array = da.from_array(r, chunks=r.chunks)
        coords = [DataArray(np.arange(array.shape[idx]) * self.resolution[k], dims=k, attrs={'units': 'nm'}) for idx,k in enumerate(self.resolution)]
        return DataArray(array, coords=coords, attrs={'name': self.content, **self.displaySettings})


sources = {'jrc_hela-4':[Source(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/10. HeLa_mitotic_17-7_Cell4_4x4x4nm/HeLa_mitotic_17-7_17_Cell4 4x4x4nm 16bit.mrc',
                             content='Raw EM Data',
                             destPath = 'fibsem/aligned',
                             resolution = {'z': base_res * 1.07, **yx_res},
                             displaySettings=display_attrs(0.26, 0.4, invertColormap=True), 
                             computePyramid=False),],
    
    'jrc_mus-pancreas-1' : [Source(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/1. Pancreas Islet_G64-2-1-HighGlucose_4x4x4nm/G64-2-1_HighGlucose 4x4x4nm 16bit.mrc',
                            content='Raw EM Data',
                            destPath = 'fibsem/aligned',
                            resolution={'z': base_res * 0.85, **yx_res},
                            displaySettings=display_attrs(0.1, 0.22, invertColormap=True),
                            computePyramid=False),],
                                     
    'jrc_hela-2' : [Source(sourcePath='/groups/cosem/cosem/data/HeLa_Cell2_4x4x4nm/Aubrey_17-7_17_Cell2 4x4x4nm.mrc',
                             content='Raw EM Data',
                             destPath = 'fibsem/aligned_v0',
                             resolution= {'z': base_res * 1.31, **yx_res},
                             displaySettings=display_attrs(0.415, .716, invertColormap=False),
                             computePyramid=False,
                             mutation="reverse_y"),

                    Source(sourcePath='/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/HeLa2.n5/mito',
                             content='Mitochondria segmentation',
                             destPath = 'predictions/mito/seg',
                             resolution = {'z': base_res * 1.31, **yx_res},
                             displaySettings=display_attrs(color='green'), 
                             computePyramid=False,
                             ancestor='fibsem/aligned_v0'),

                    Source(sourcePath='/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/HeLa2.n5/plasma_membrane',                             
                             content='Plasma membrane segmentation',
                             destPath = 'predictions/plasma_membrane/seg',
                             resolution = {'z': base_res * 1.31, **yx_res},
                             displaySettings=display_attrs(color='orange'), 
                             computePyramid=False,
                             ancestor='fibsem/aligned_v0'),
                             
                    Source(sourcePath='/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/HeLa2.n5/er',
                             content='Endoplasmic reticulum segmentation',
                             destPath = 'predictions/er/seg',
                             resolution = {'z': base_res * 1.31, **yx_res},
                             displaySettings=display_attrs(color='blue'), 
                             computePyramid=False,
                             ancestor='fibsem/aligned_v0'),

                    Source(sourcePath='/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/HeLa2.n5/MVB',
                             content='Multivesicular body segmentation',
                             destPath = 'predictions/mvb/seg',
                             resolution = {'z': base_res * 1.31, **yx_res},
                             displaySettings=display_attrs(color='magenta'), 
                             computePyramid=False,
                             ancestor='fibsem/aligned_v0'),

                    Source(sourcePath='/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/HeLa2.n5/vesicle',
                             content='Vesicle segmentation',
                             destPath = 'predictions/vesicle/seg',
                             resolution = {'z': base_res * 1.31, **yx_res},
                             displaySettings=display_attrs(color='red'), 
                             computePyramid=False,
                             ancestor='fibsem/aligned_v0'),
                             
                    Source(sourcePath='/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/HeLa2.n5/golgi',
                             content='Golgi segmentation',
                             destPath = 'predictions/golgi/seg',
                             resolution = {'z': base_res * 1.31, **yx_res},
                             displaySettings=display_attrs(color='cyan'), 
                             computePyramid=False,
                             ancestor='fibsem/aligned_v0')         
                             ],
    
    'jrc_hela-3' : [Source(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/3. HeLa3_Aubrey_17-7_17_Cell3_4x4x4nm/HeLa_Cell3_17-7_17 4x4x4nm 16bit.mrc',
                             content='Raw EM Data',
                             destPath = 'fibsem/aligned',
                             resolution= {'z': base_res * 0.81, **yx_res},
                             displaySettings=display_attrs(0.25, 0.55, invertColormap=True),
                             computePyramid=False)],
                             
    'jrc_jurkat-1' : Source(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/7. Jurkat_Cryo_2017_FS96_Cell1_4x4x4nm/Jurkat_Cryo_2017_FS96_Area1 4x4x4nm 16bit.mrc',
                            content='Raw EM Data',
                            destPath = 'fibsem/aligned',
                            resolution={'z': base_res * 0.86, **yx_res},
                            displaySettings=display_attrs(0.295, 0.457, invertColormap=True),
                            computePyramid=True),
                             
    'jrc_macrophage-2': Source(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/6. Macrophage_FS80_Cell2_4x4x4nm/Macrophage_FS80_Cell2 4x4x4nm 16bit.mrc',
                                content='Raw EM Data',
                                destPath='fibsem/aligned',
                                resolution={'z': base_res * 0.84, **yx_res},
                                displaySettings=display_attrs(0.295, 0.457, invertColormap=True),
                                 computePyramid=True),
                             
    'jrc_sum159-1': Source(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/8. SUM159_WT45_Cell2_4x4x4nm/SUM159_WT45_Cell2_Cryo_20171009 4x4x4nm 16bit.mrc',
                            content='Raw EM Data',
                            destPath='fibsem/aligned',
                            resolution={'z': base_res * 1.14, **yx_res},
                              displaySettings=display_attrs(0.224, 0.363, invertColormap=True),
                             computePyramid=False),
                             
    'jrc_ctl-id8-1': Source(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/9. TCell_on_cancer_4x4x4nm/TCell_on_cancer_Atlas1 4x4x4nm 16bit.mrc',
                              content='Raw EM Data',
                              destPath = 'fibsem/aligned',
                              resolution={'z': base_res * 0.87, **yx_res},
                              displaySettings=display_attrs(0.157, 0.292, invertColormap=True),
                              computePyramid=False),
    
    'jrc_fly-fsb-1' : Source(sourcePath='/groups/hess/hesslab/HighResPaper_rawdata/4. Fly Fan Shaped Body[Column1-9]_Z0519-11_4x4x4nm/FB-Z0519-11 4x4x4nm 16bit.mrc',
                                content='Raw EM Data',
                                destPath = 'fibsem/aligned',
                                resolution= {'z': base_res * 1.0, **yx_res},
                                displaySettings=display_attrs(0.027, 0.069, invertColormap=True),
                                computePyramid=False),
                             
    'jrc_fly-acc-calyx-1' : Source(sourcePath='/nrs/hess/for Davis/AceeCalyx_Z0519-15 4x4x4nm 16bit.mrc',
                                    content='Raw EM Data',
                                    destPath = 'fibsem/aligned',
                                    resolution={'z': base_res * 0.93, **yx_res},
                                    displaySettings=display_attrs(0.023, 0.051, invertColormap=True),
                                    computePyramid=False),
}