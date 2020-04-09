import dask
import numpy as np
from xarray import DataArray
import dask.array as da
from fst.io import read, access
from dataclasses import dataclass
from pathlib import Path
from fst.pyramid import lazy_pyramid
from skimage.exposure import rescale_intensity as rescale
from numcodecs import GZip
from fst.io import create_arrays
from typing import List, Tuple, Union, Dict

output_chunks = (128,128,128)
output_dtype = 'uint8'
output_multiscale_format = 'neuroglancer_n5'
n5_attr_axes = {'z': 2, 'y': 1, 'x': 0}
zr_axes = {'z': 0, 'y': 1, 'x': 2}
# output_downscale = {'z': 2, 'y': 2, 'x': 2}
# output_path = Path('/nrs/cosem/davis/s3_testing/')

def name_prediction_array(path: Union[str, Path]):
    from fst.io.io import split_path_at_suffix
    container, array = split_path_at_suffix(path)
    name = str(array.parts[-1])
    iterations = container.parts[-1].split('_it')[-1].split('.')[0]
    setup = container.parts[-3]
    if setup.find('setup') == -1: 
        raise ValueError(f'could not find an instance of "setup" in {setup}')
    result = f'{name}_{setup}_it{iterations}'
    return result

def group_attrs(pyramids: list) -> dict:
    scales = [list(s.scale_factors) for s in pyramids]
    units = list(pyramids[0].units.values())
    axes = list(pyramids[0].dims)
    arr_attrs = [array_attrs(p) for p in pyramids]
    pixelResolution = {'pixelResolution': arr_attrs[0]['pixelResolution']}
    
    n5v_attrs = n5v_multiscale_group_attrs(scales, pixelResolution=pixelResolution)
    neuroglancer_attrs = neuroglancer_multiscale_group_attrs(axes=axes, units = units)    
    cosem_attrs = cosem_multiscale_group_attrs(transforms=[a['transform'] for a in arr_attrs], name=arr_attrs[0]['name'])
    
    return {**n5v_attrs, **neuroglancer_attrs, **cosem_attrs}

def array_attrs(arr) -> dict:
    translate = [float(k.data[0]) for k in arr.coords.values()]
    scale = list(map(float, np.subtract([k.data[1] for k in arr.coords.values()], translate).tolist()))
    units = list(arr.attrs['units'].values())
    name = str(Path(arr.attrs['path']).parts[-1])
    cosem_attrs = cosem_array_attrs(translate=translate, scale=scale, units=units, name=name)
    n5v_attrs = n5v_array_attrs(dimensions=scale, unit=units[0])
    return {**cosem_attrs, **n5v_attrs}

def neuroglancer_multiscale_group_attrs(axes: List, units: List) -> dict:
    # see https://github.com/google/neuroglancer/issues/176#issuecomment-553027775
    return {'axes': axes, 'units': units}

def cosem_array_attrs(translate: List, scale: List, units: List, name: str) -> dict:
    return {'name': name,'transform': {'translate': translate, 'scale': scale, 'units': units}}

def cosem_multiscale_group_attrs(transforms: List, name: str) -> dict:
    return {'name': name, 'multiscale': [{'path': f'./s{idx}', 'transform': t} for idx,t in enumerate(transforms)]}

def n5v_array_attrs(dimensions: List, unit: str) -> dict:
    return {'pixelResolution': {'dimensions' : dimensions, 'unit': unit}}

def n5v_multiscale_group_attrs(scales: List[List], pixelResolution: dict) -> dict:
    return {'scales': scales, **pixelResolution}

def DataArrayFactory(source_path: Union[str, Path], dest_path: Union[str, Path], chunks: Tuple[int]):
    arr = read(source_path) 
    try:
        n5_scale = arr.attrs['pixelResolution']['dimensions']
        n5_unit = arr.attrs['pixelResolution']['unit']
    except KeyError:
        n5_scale = arr.attrs['resolution']
        n5_unit='nm'
    scale = {k:n5_scale[v] for k,v in n5_attr_axes.items()}
    coords = {k: np.arange(arr.shape[v]) * scale[k] for k,v in zr_axes.items()}
    units = {k: n5_unit for k in n5_attr_axes}
    data = DataArray(da.from_array(arr, chunks=chunks), coords=tuple(coords.values()), dims=coords.keys())
    data.attrs.update({'units': units, 'source': str(source_path), 'path': str(dest_path)})
    
    return data

def save_blockwise(arr, path: Union[str, Path], block_info):
    from fst.io import access    
    import time
    import dask
    # workers create their own client object that doesn't use the same config
    dask.config.set({'distributed.comm.timeouts.connect':'160s'})
    pos = block_info[0]["array-location"]    
    idx = tuple(slice(*i) for i in pos)
    num_retries = 3
    sleepdur = .5
    retval = 1
    # this idiotic construct is in here to mitigate a possible race condition when writing n5 containers via zarr
    # basically sometimes there's an OSError when writing to an array; i think this might be caused by the directory structure 
    # changing between access and write, as could happen when many workers are writing data. 
    for r in range(num_retries):
        if retval == 0: break
        try:
            sink = access(path, mode="a")
            sink[idx] = arr
            success = True
            retval = 0
        except OSError:
            time.sleep(sleepdur)
        
    return np.expand_dims(retval, tuple(range(arr.ndim))).astype(arr.dtype)


@dataclass
class dataset:
    name: str
    sources: Dict[str, DataArray]    

def prepare_store(dataset: dataset, output_path: Union[str, Path], output_downscale: dict):
    paths = list(dataset.sources.keys())
    to_store = []
    root_attrs = {'multiscale_data': paths}
    root_container_path = Path(output_path) / f'{dataset.name}.n5'
    access(root_container_path, mode='a').attrs.put(root_attrs)
    for group_path, array in dataset.sources.items():
        pyr = lazy_pyramid(array, np.mean, list(output_downscale.values()))
        grp_attrs = group_attrs(pyr)
        names, shapes, dtypes, compressors, chunks, arr_attrs = [],[],[],[],[],[]
        for ind_p, p in enumerate(pyr):           
            names.append(f's{ind_p}') 
            shapes.append(p.shape) 
            dtypes.append(p.dtype)
            compressors.append(GZip(-1))
            chunks.append(output_chunks) 
            arr_attrs.append(array_attrs(p))
        
        multiscale_grp = root_container_path / group_path
        grp = create_arrays(multiscale_grp, 
                        names=names, 
                        shapes=shapes,
                        dtypes=dtypes,
                        compressors=compressors,
                        chunks=chunks,
                        group_attrs=grp_attrs,
                        array_attrs=arr_attrs)

        pyr_save = []
        for idx, p in enumerate(pyr):
            level_path = multiscale_grp / names[idx]    
            o_chunks = np.maximum(read(level_path).chunks, p.data.chunksize)
            pyr_save.append(p.data.rechunk(o_chunks).map_blocks(save_blockwise, path=level_path, dtype=p.data.dtype))
        to_store.append(pyr_save)

    return to_store
