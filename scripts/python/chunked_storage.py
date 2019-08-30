import dask.array as da
import argparse
from glob import glob
from pathlib import Path
from fst.io.ingest import padstack, arrays_from_delayed
from fst.io import read
import numpy as np
from typing import Union
import logging
import sys
import zarr
import numcodecs
from dask.diagnostics import ProgressBar
from fst.distributed import bsub_available
from dask.utils import SerializableLock
from distributed import Client
import time

OUTPUT_FMTS = {"n5"}
max_chunksize = 1024
compressor = numcodecs.GZip(level=1)
lock = False
# todo: move data writing functions to fst.io


def save_blockwise(v, store_path, block_info):
    # assumes input is 5D with first dimension == 1 and 
    # second dimension == 2, and third + fourth dimensions the full size
    # of the zarr/n5 file located at store_path
    split_dim = 1
    sinks = [zarr.open(store_path,mode='w')[f'/volumes/raw/ch{d}'] for d in range(2)]
    
    pos = block_info[0]['array-location']
    # get rid of the channel dimension
    pos.pop(split_dim)
    idx = tuple(slice(*i) for i in pos)
    for ind in range(2):
        sinks[ind][idx] = np.expand_dims(v[0,ind], 0)
    return np.zeros((1,) * v.ndim)


def n5_store(
    array: Union[da.Array, list],
    metadata: list,
    dest: str,
    path: str = "/volumes",
    name: Union[str, list] = "raw",
) -> da.Array:
    store = zarr.N5Store(dest)
    group = zarr.group(overwrite=True, store=store, path=path)

    # check that array is a list iff name is a list
    assert isinstance(array, list) == isinstance(name, list)

    if isinstance(array, list):
        result = []
        for ind, arr in enumerate(array):
            z = group.zeros(
                shape=arr.shape,
                chunks=arr.chunksize,
                dtype=arr.dtype,
                compressor=compressor,
                name=name[ind],
            )
            z.attrs["metadata"] = metadata
            result.append(arr.store(z, compute=False, lock=lock))

    elif isinstance(array, da.Array):
        z = group.zeros(
            shape=array.shape,
            chunks=array.chunksize,
            dtype=array.dtype,
            compressor=compressor,
            name=name,
        )
        z.attrs["metadata"] = metadata
        result = array.store(z, compute=False, lock=lock)
    return result

stores = dict(n5=n5_store)


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
    #stacked = padstack(data, constant_values="minimum-minus-one").swapaxes(0, 1)
    stacked = padstack(data, constant_values="minimum-minus-one")
    rechunked = stacked
    # For raw data, single chunks along channel and z axes. Channels will become their own datasets
    #rechunked = stacked.rechunk((1,1,*np.where(np.array(stacked.shape[2:]) < max_chunksize, stacked.shape[2:], max_chunksize,),))
    logging.info(
        f"Assembled dataset with shape {rechunked.shape} using chunk sizes {rechunked.chunksize}"
    )

    return rechunked, meta


def save_data(data: Union[da.Array, list]):
    with ProgressBar():
        da.compute(data)


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
        default=None
    )

    parser.add_argument(
        "--dry-run",
        help="Dry run. Does everything except save data.",
        action="store_true",
    )

    parser.add_argument(
        "-lf",
        "--log_file",
        help="Path to a logfile that will be created to track progress of the conversion."
    )

    args = parser.parse_args()
    #lock = SerializableLock()

    logging.basicConfig(filename=args.log_file,
                        filemode='a',
                        level=logging.INFO)

    num_workers = int(args.num_workers)
    output_fmt = Path(args.dest).suffix[1:]
    if output_fmt not in OUTPUT_FMTS:
        raise NotImplementedError(
            f"Cannot write a chunked store using format {output_fmt}. Try one of {OUTPUT_FMTS}"
        )

    padded_array, metadata = prepare_data(args.source, max_chunksize=max_chunksize)
    dataset_path = '/volumes/raw'
    dataset_names = ['ch' + str(r) for r in range(2)]
    #my_stores = stores[output_fmt]([*padded_array], metadata, args.dest, path=dataset_path, name=dataset_names)
    
    # stick this in a function
    group = zarr.group(overwrite=True, store=zarr.N5Store(args.dest), path=dataset_path)
    # we are sending elements along the first axis to separate arrays 
    out_shape = list(padded_array.shape)
    out_shape.pop(1)
    [group.zeros(name=n, shape=out_shape, compressor=compressor, chunks=(1,max_chunksize,max_chunksize)) for n in dataset_names]
    import pdb; pdb.set_trace()
    if not args.dry_run:
        running_on_cluster = bsub_available()
        if running_on_cluster:
            from fst.distributed import get_jobqueue_cluster
            cluster = get_jobqueue_cluster(project='cosem')
            client = Client(cluster)
            cluster.start_workers(num_workers)
        else:
            from distributed import LocalCluster
            cluster = LocalCluster(n_workers=num_workers)
            client = Client(cluster)

        logging.info(f'Begin saving data to {args.dest}. View status at the following address: {cluster.dashboard_link}')
        padded_array.map_blocks(save_blockwise, store_path=args.dest, dtype=padded_array.dtype).compute()
        # save_data(my_stores)
        elapsed_time = time.time() - start_time
        logging.info(f'Save completed in {elapsed_time} s')
