import argparse
from glob import glob
from pathlib import Path
from fst.io.ingest import padstack, arrays_from_delayed
from fst.io import read, get_umask, chmodr
import numpy as np
from typing import Union
import logging
import zarr
import numcodecs
from fst.distributed import bsub_available
from distributed import Client
import time

OUTPUT_FMTS = {"n5"}
max_chunksize = 1024
compressor = numcodecs.GZip(level=9)
# raw data are stored z c y x, we will split images into two channels along the channel dimension
channel_dim = 1


def prepare_data(path: Union[str, list], max_chunksize: int) -> tuple:
    if isinstance(path, str):
        fnames = sorted(glob(path))
    elif isinstance(path, list):
        fnames = path
    else:
        raise ValueError(f"Path variable should be string or list, not {type(path)}")

    logging.info(f"Preparing {len(fnames)} images...")
    data_eager = read(fnames, lazy=False)
    meta, shapes, dtypes = [], [], []
    # build lists of metadata for each image
    for d in data_eager:
        meta.append(d.header.__dict__)
        shapes.append(d.shape)
        dtypes.append(d.dtype)
    data = arrays_from_delayed(read(fnames, lazy=True), shapes=shapes, dtypes=dtypes)
    stacked = padstack(data, constant_values="minimum-minus-one")
    logging.info(
        f"Assembled dataset with shape {stacked.shape} using chunk sizes {stacked.chunksize}"
    )

    return stacked, meta


def prepare_chunked_store(
    dest_path, data_path, names, shapes, dtypes, compressor, chunks, metadata
):
    group = zarr.group(overwrite=True, store=zarr.N5Store(dest_path), path=data_path)
    for n, s, d, c, m in zip(names, shapes, dtypes, chunks, metadata):
        g = group.zeros(name=n, shape=s, dtype=d, compressor=compressor, chunks=c)
        g.attrs["metadata"] = m


def save_blockwise(v, store_path, split_dim, block_info):
    # assumes input is 5D with first dimension == 1 and
    # second dimension == 2, and third + fourth dimensions the full size
    # of the zarr/n5 file located at store_path

    num_sinks = v.shape[split_dim]
    sinks = [
        zarr.open(store_path, mode="a")[f"/volumes/raw/ch{d}"] for d in range(num_sinks)
    ]

    pos = block_info[0]["array-location"]
    # get rid of the split dimension
    pos.pop(split_dim)
    idx = tuple(slice(*i) for i in pos)
    for ind in range(num_sinks):
        sinks[ind][idx] = np.expand_dims(v[0, ind], 0)
    return np.zeros((1,) * v.ndim)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Save a sequence of images to a chunked store."
    )
    parser.add_argument(
        "-s",
        "--source",
        help="Files to process. Must be either a single directory (e.g., `/data/` "
        "or a wild-card expansion of a single directory (e.g., `/data/*.dat`). "
        "Files will be sorted by filename.",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-d",
        "--dest",
        help="The chunked store to create with the input files. Supported chunked store"
        f" formats: {OUTPUT_FMTS} formats are supported",
        required=True,
    )

    parser.add_argument(
        "-nw",
        "--num_workers",
        help="The number of workers to use for distributed computation",
        default=8,
    )

    parser.add_argument(
        "--dry-run",
        help="Dry run. Does everything except save data.",
        action="store_true",
    )

    parser.add_argument(
        "-lf",
        "--log_file",
        help="Path to a logfile that will be created to track progress of the conversion.",
    )

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, filemode="a", level=logging.INFO)

    num_workers = int(args.num_workers)

    output_fmt = Path(args.dest).suffix[1:]
    if output_fmt not in OUTPUT_FMTS:
        raise NotImplementedError(
            f"Cannot write a chunked store using format {output_fmt}. Try one of {OUTPUT_FMTS}"
        )

    padded_array, metadata = prepare_data(args.source, max_chunksize=max_chunksize)
    dataset_path = "/volumes/raw"
    num_channels = padded_array.shape[channel_dim]
    n5_names = [f"ch{r}" for r in range(num_channels)]
    n5_shapes = list(padded_array.shape)
    n5_shapes.pop(channel_dim)
    n5_shapes = num_channels * (n5_shapes,)
    n5_chunks = num_channels * ((1, max_chunksize, max_chunksize),)
    n5_dtypes = num_channels * (padded_array.dtype,)

    logging.info('Initializing chunked storage....')
    prepare_chunked_store(
        dest_path=args.dest,
        data_path=dataset_path,
        names=n5_names,
        shapes=n5_shapes,
        dtypes=n5_dtypes,
        compressor=compressor,
        chunks=n5_chunks,
        metadata=metadata,
    )

    if not args.dry_run:
        running_on_cluster = bsub_available()
        if running_on_cluster:
            from fst.distributed import get_jobqueue_cluster

            cluster = get_jobqueue_cluster(project="cosem")
            cluster.start_workers(num_workers)
        else:
            from distributed import LocalCluster
            cluster = LocalCluster(n_workers=num_workers)
            client = Client(cluster)

        client = Client(cluster)
        logging.info(
            f"Begin saving data to {args.dest}. View status at the following address: {cluster.dashboard_link}"
        )
        padded_array.map_blocks(
            save_blockwise,
            store_path=args.dest,
            split_dim=channel_dim,
            dtype=padded_array.dtype,
        ).compute()
        # zarr saves data in temporary files, which have very
        # restricted permissions. This function call recursively applies
        # new permissions to all the files  in the newly created container based on the current umask setting

        client.close()
        cluster.close()

        logging.info(f'Updating permissions of files in {args.dest}')
        chmodr(args.dest, mode='umask')
        
        elapsed_time = time.time() - start_time
        logging.info(f"Save completed in {elapsed_time} s")
