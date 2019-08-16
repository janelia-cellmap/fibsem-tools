import dask.array as da
import argparse
from glob import glob
from pathlib import Path
from fst.ingest import padstack
from fst import readfibsem
import numpy as np
from typing import Union, NoReturn
import logging
import sys
import zarr
import numcodecs
from multiprocessing import cpu_count
from dask.diagnostics import ProgressBar

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
_INPUT_FMTS = {"dat"}
_OUTPUT_FMTS = {"zarr", "n5"}
max_chunksize = 256
readers = dict(dat=readfibsem)

roi_size = 1024


def n5_store(
    array: da.Array, dest: str, path: str = "/volumes", name: str = "raw"
) -> da.Array:
    store = zarr.N5Store(dest)
    compressor = numcodecs.GZip(level=1)
    group = zarr.group(overwrite=True, store=store, path=path)

    z = group.zeros(
        shape=array.shape,
        chunks=array.chunksize,
        dtype=array.dtype,
        compressor=compressor,
        name=name,
    )

    return array.store(z, compute=False)


def zarr_store(
    array: da.Array, dest: str, path: str = "/volumes", name: str = "raw"
) -> da.Array:
    compressor = numcodecs.GZip(level=1)
    group = zarr.group(overwrite=True, store=dest, path=path)

    z = group.zeros(
        shape=array.shape,
        chunks=array.chunksize,
        dtype=array.dtype,
        compressor=compressor,
        name=name,
    )

    return array.store(z, compute=False)


stores = dict(n5=n5_store, zarr=zarr_store)


def prepare_data(
    path: Union[str, list], max_chunksize: int = max_chunksize
) -> da.Array:
    if isinstance(path, str):
        fnames = sorted(glob(path))
    elif isinstance(path, list):
        fnames = path
    else:
        raise ValueError(f"Path variable should be string or list, not {type(path)}")
    input_fmt = Path(fnames[0]).suffix[1:]
    if input_fmt not in _INPUT_FMTS:
        raise ValueError(
            f"Cannot load images with format {input_fmt}. Try {_INPUT_FMTS} instead."
        )
    logging.info(f"Preparing {len(fnames)} images...")
    stacked = padstack(readers[input_fmt](fnames)).swapaxes(0, 1)
    logging.info(f"Assembled dataset with shape {stacked.shape}")
    rechunked = stacked.rechunk(
        (
            1,
            *np.where(
                np.array(stacked.shape[1:]) < max_chunksize,
                stacked.shape[1:],
                max_chunksize,
            ),
        )
    )
    return rechunked


def save_data(data: da.Array, dest: str):
    logging.info(f"Begin saving data to {dest}")
    num_workers = max(int(cpu_count() / 8), 2)
    with ProgressBar():
        data.compute(num_workers=num_workers)


if __name__ == "__main__":
    try:
        import zarr
    except ImportError:
        _OUTPUT_FMTS.remove("zarr")
        _OUTPUT_FMTS.remove("n5")
    try:
        import h5py
    except ImportError:
        _OUTPUT_FMTS.remove("hdf5")
    if len(_OUTPUT_FMTS) == 0:
        raise ImportError(f"No chunked storage library found. Tried {_OUTPUT_FMTS}")

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
        f" formats: {_OUTPUT_FMTS} formats are supported",
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        help="Dry run. Does everything except save data.",
        action="store_true",
    )

    args = parser.parse_args()

    output_fmt = Path(args.dest).suffix[1:]
    if output_fmt not in _OUTPUT_FMTS:
        raise NotImplementedError(
            f"Cannot write a chunked store using format {output_fmt}. Try one of {_OUTPUT_FMTS}"
        )

    padded_array = prepare_data(args.source)
    store = stores[output_fmt](padded_array, args.dest)
    if not args.dry_run:
        save_data(store, args.dest)
