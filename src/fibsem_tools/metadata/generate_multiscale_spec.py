import click
from numpy import dtype
import zarr
from fibsem_tools.metadata.multiscale_generation import (
    ClusterSpec,
    DownsamplingSpec,
    ReadableArrayStore,
    WriteableArrayStore,
    WorkerSpec,
    MultiscaleStorageSpec,
)


def generate_spec(json_path: str, create_source: bool):
    source_root = "/groups/cosem/cosem/bennettd/scratch/test.zarr"
    source_component = "source"
    dest_component = "dest"
    rank = 3
    if create_source:
        source_data = zarr.open(
            store=zarr.NestedDirectoryStore(source_root),
            path=source_component,
            mode="w",
            shape=(2048,) * rank,
            chunks=(64,) * rank,
            dtype="uint8",
        )
        source_data[:] = 1
    source = ReadableArrayStore(
        url=f"{source_root}/{source_component}",
        storage_options={},
        chunks=(128,) * rank,
    )

    dest = WriteableArrayStore(
        url=f"{source_root}/{dest_component}",
        storage_options={},
        chunks=(64,) * rank,
        access_mode=("w", "w"),
    )

    downsampling_spec = DownsamplingSpec(
        method="mean", factors=(2,) * rank, levels=(0, 1, 3, 4), chunks=(128,) * rank
    )

    cluster_spec = ClusterSpec(
        deployment="dask_lsf",
        worker=WorkerSpec(num_workers=1, num_cores=10, memory="15GB"),
    )

    spec = MultiscaleStorageSpec(
        source=source,
        destination=dest,
        downsampling_spec=downsampling_spec,
        cluster_spec=cluster_spec,
        logging_dir="/groups/scicompsoft/home/bennettd/logs",
    )

    with open(json_path, mode="w") as fh:
        fh.write(spec.json(indent=2))


@click.command()
@click.argument("json_path", type=str)
@click.option("--create_source", type=bool)
def main(json_path: str, create_source: bool = False):
    generate_spec(json_path, create_source)


if __name__ == "__main__":
    main()
