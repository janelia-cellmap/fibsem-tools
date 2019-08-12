import yaml
import io
from glob import glob
from pathlib import Path
from os.path import exists

def prepare_experiment_dicts(rootdir):
    dsets = sorted(glob(rootdir + '*/*.n5'))
    keys = [Path(d).parts[-1] for d in dsets]
    resolution = [get_resolution_from_path(d) for d in dsets]
    resolution_dicts = [{'x': r[0],'y' : r[1],'z': r[2]} for r in resolution]
    needs_mask = len(dsets) * [False]
    mask_threshold = len(dsets) * [0.0]
    mask_dicts = [{'needs_mask' : needs_mask[ind], 'mask_threshold' : mask_threshold[ind]} for ind in range(len(dsets))]    
    data = []
    for ind in range(len(dsets)):
        exp = {'path': dsets[ind], 
                'resolution' : resolution_dicts[ind], 
                'mask_params' : mask_dicts[ind]}
        data.append(exp)
        
    return data

def get_resolution_from_string(path):
    # get the resolution from the path. resolution takes the form 8x8x8nm
    import re    
    res = None
    match = re.search(r'[0-99]x[1-99]x[1-99]', path)    
    if match:
        res = re.findall(r'[1-99]', match.group())
        res = [int(r) for r in res]
    return res

def get_resolution_from_path(path, depth=2):
    # break a path into parts and check for resolution in each part of the path
    from pathlib import Path
    parts = Path(path).parts
    res = None
    for level in range(1, depth+1):
        part = parts[-level]
        res = get_resolution_from_string(part)
        if res is not None:
            break
    return res

if __name__=='__main__':
    rootdir ='/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/' 
    metadata_fname = '/groups/hess/hess_collaborators/scripts/params.yaml'
    overwrite = False
    data = prepare_experiment_dicts(rootdir)
    print(f'Found {len(data)} datasets.')
    for d in data:
        print(d)
    
    if exists(metadata_fname):
        if overwrite == True:
            print(f'File {metadata_fname} exists. Overwriting.')
        else:
            print(f'File {metadata_fname} exists. Set `overwrite=True` to overwrite it.')

    # Write YAML file
    with io.open(metadata_fname, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True, sort_keys=False)   
