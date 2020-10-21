import s3fs
from dask import bag
import click
from pathlib import Path
from dask.diagnostics import ProgressBar
from typing import Sequence, Optional, Tuple, List
from fst.io import fwalk
import os

STAGES = ("dev", "prod", "val")


def iterput(
    sources: Sequence[str],
    dests: Sequence[str],
    tags: Sequence[Optional[dict]],
    profile: Optional[str] = None,
):
    """
    Given a sequence of sources, dests, and tags, save each source to dest with a tag.    
    """
    fs = s3fs.S3FileSystem(profile=profile)
    for source, dest, tag in zip(sources, dests, tags):
        fs.put(source, dest)
        if tag is not None:
            fs.put_tags(dest, tag)
    return True


def s3put(
    dest_root: str,
    source_path: str,
    endswith: Optional[str] = "",
    tags: Optional[dict] = None,
    partition_size=None,
):
    sources = tuple(map(str, fwalk(source_path, endswith)))
    dests = tuple(
        str(Path(dest_root) / Path(f).relative_to(source_path)) for f in sources
    )

    source_bag = bag.from_sequence(sources, partition_size=partition_size)
    dest_bag = bag.from_sequence(dests, partition_size=partition_size)
    tag_bag = bag.from_sequence((tags,) * len(sources), partition_size=partition_size)

    return bag.map_partitions(iterput, source_bag, dest_bag, tag_bag)


@click.command()
@click.argument("source_paths", required=True, nargs=-1)
@click.option("-b", "--bucket", required=True, type=str)
@click.option("-ew", "--endswith", required=True, type=str)
@click.option("-vt", "--version-tag", required=True, type=str)
@click.option("-st", "--stage-tag", required=True, type=str)
@click.option("-dvt", "--developer-tag", required=True, type=str)
@click.option("-pt", "--project-tag", required=True, type=str)
@click.option("-dt", "--description-tag", required=True, type=str)
def s3put_cli(
    source_paths,
    bucket,
    endswith,
    version_tag,
    stage_tag,
    developer_tag,
    project_tag,
    description_tag,
):
    for source_path in source_paths:
        dest_root = Path(bucket) / Path(source_path).stem
        assert stage_tag in STAGES
        tags = {
            "VERSION": version_tag,
            "DEVELOPER": developer_tag,
            "STAGE": stage_tag,
            "PROJECT": project_tag,
            "DESCRIPTION": description_tag,
        }
        s3put(dest_root, source_path, endswith=endswith, tags=tags).compute()
    return 0


if __name__ == "__main__":
    pbar = ProgressBar()
    pbar.register()
    s3put_cli()
