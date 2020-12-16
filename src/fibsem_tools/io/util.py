from dask import delayed, bag
from shutil import rmtree
from glob import glob
import os
from typing import Sequence, List
import toolz as tz
from typing import Sequence, Union, Any, Tuple
from pathlib import Path

@delayed
def _rmtree_after_delete_files(path: str, dependency: Any):
    rmtree(path)


def rmtree_parallel(path: Union[str, Path], branch_depth: int=1, compute: bool=True):
    branches = glob(os.path.join(path, *('*',) * branch_depth))
    deleter = os.remove
    files = list_files_parallel(branches)
    deleted_files = bag.from_sequence(files).map(deleter)
    result = _rmtree_after_delete_files(path, dependency=deleted_files)
    
    if compute:
        return result.compute(scheduler='threads')
    else:
        return result


def list_files(paths: Union[Sequence[Union[str, Path]], str, Path]):
    if isinstance(paths, str) or isinstance(paths, Path):
        if os.path.isdir(paths):
            return list(tz.concat((os.path.join(dp, f) for f in fn) for dp, dn, fn in os.walk(paths)))
        elif os.path.isfile(paths):
            return [paths]
        else:
            raise ValueError(f'Input argument {paths} is not a path or a directory')
    
    elif isinstance(paths, Sequence):
        sortd = sorted(paths, key=os.path.isdir)
        files, dirs = tuple(tz.partitionby(os.path.isdir, sortd))
        return list(tz.concatv(files, *tz.map(list_files, dirs)))


def list_files_parallel(paths: Union[Sequence[Union[str, Path]], str, Path], compute: bool=True):
    result = []
    delf = delayed(list_files)
    
    if isinstance(paths, str) or isinstance(paths, Path):
        result = bag.from_delayed([delf(paths)])
    elif isinstance(paths, Sequence):
        result = bag.from_delayed([delf(p) for p in paths])
    else:
        raise TypeError(f'Input must be a string or a sequence, not {type(paths)}')
    
    if compute:
        return result.compute(scheduler='threads')
    else:
        return result


def split_path_at_suffix(suffixes: Tuple[str],
    upper_path: Union[str, Path], lower_path: Union[str, Path] = "") -> List[Path]:
    """
    Recursively climb a path, checking at each level of the path whether the tail of the path represents a directory
    with a container extension. Returns the path broken at the level where a container is found.  
    """
    upper, lower = Path(upper_path), Path(lower_path)

    if upper.suffix in suffixes:
        result = [upper, lower]
    else:
        if len(upper.parts) >= 2:
            result = split_path_at_suffix(suffixes,
                Path(*upper.parts[:-1]), Path(upper.parts[-1], lower)
            )
        else:
            raise ValueError(
                f"Could not find any suffixes matching {suffixes} in {upper / lower}"
            )

    return result