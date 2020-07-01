import s3fs
from dask import bag
import click 
from pathlib import Path
from dask.diagnostics import ProgressBar
from typing import Sequence, Optional, Tuple, List
import os 

STAGES = ('dev', 'prod', 'val')

def fwalk(source: str, endswith='') -> List[str]:
    """
    Use os.walk to recursively parse a directory tree, returning a list containing the full paths
    to all files with filenames ending with `endswith`.
    """
    results = []
    for p, d, f in os.walk(source):
        for file in f:
            if file.endswith(endswith):
                results.append(os.path.join(p, file))
    return results

def iterput(sources: Sequence[str], dests: Sequence[str], tags: Sequence[Optional[dict]], profile: Optional[str]=None):
    """
    Given a sequence of (source, dest) pairs, copy the file at `source` to the target at `dest`.    
    """           
    fs = s3fs.S3FileSystem(profile=profile)
    for source, dest, tag in zip(sources, dests, tags):
        fs.put(source, dest)
        if tag is not None:
            fs.put_tags(dest, tag)
    return True        

def s3put(dest_root: str, source_path: str, endswith: Optional[str]='', profile=None, tags: Optional[dict]=None, **kwargs):    
    sources = fwalk(source_path, endswith)
    dests = tuple(dest_root / Path(f).relative_to(source_path) for f in sources)
    
    source_bag = bag.from_sequence(sources)
    dest_bag = bag.from_sequence(dests)
    tag_bag = bag.from_sequence((tags,) * len(sources))

    return bag.map_partitions(iterput, source_bag, dest_bag, tag_bag)

@click.command()
@click.option('-b', '--bucket', required=True, type=str)
@click.option('-s', '--source-path', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('-ew', '--endswith', required=True, type=str)
@click.option('-vt', '--version-tag', required=True, type=str)
@click.option('-st', '--stage-tag', required=True, type=str)
@click.option('-dvt', '--developer-tag', required=True, type=str)
@click.option('-pt', '--project-tag', required=True, type=str)
@click.option('-dt', '--description-tag', required=True, type=str)
def s3put_cli(bucket, source_path, endswith, version_tag, stage_tag, developer_tag, project_tag, description_tag):
    dest_root = Path(bucket) / Path(source_path).stem 
    assert stage_tag in STAGES
    tags = {'VERSION': version_tag, 
            'DEVELOPER': developer_tag, 
            'STAGE': stage_tag, 
            'PROJECT': project_tag, 
            'DESCRIPTION': description_tag}
    return s3put(dest_root, source_path, endswith=endswith, tags=tags).compute()


if __name__ == '__main__':
    pbar = ProgressBar()
    pbar.register()
    s3put_cli()
