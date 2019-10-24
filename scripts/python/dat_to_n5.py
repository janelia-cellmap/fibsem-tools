import argparse
from glob import glob
from pathlib import Path
from fst.io.ingest import padstack, arrays_from_delayed
from fst.io import read, chmodr
import numpy as np
from typing import Union
import logging
import zarr
import numcodecs
from fst.distributed import bsub_available
from distributed import Client
import time
import dask.array as da
from fst.pyramid import lazy_pyramid, get_downsampled_offset

OUTPUT_FMTS = {"n5"}
max_chunksize = 1024
compressor = numcodecs.GZip(level=-1)
# raw data are stored z c y x, we will split images into two channels along the channel dimension
channel_dim = 1
downscale_factor = 2


def prepare_data(path):
    if isinstance(path, str):
        fnames = sorted(glob(path))
    elif isinstance(path, list):
        fnames = path
    else:
        raise ValueError(f"Path variable should be string or list, not {type(path)}")
    # set the function to use for pyramid downscaling
    reduction = np.mean
    scaling_factors = (1, 1, downscale_factor, downscale_factor)
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

    pyramid = lazy_pyramid(stacked, reduction, scaling_factors, preserve_dtype=True)

    return pyramid, meta


def prepare_chunked_store(
    dest_path, data_path, names, shapes, dtypes, compressor, chunks, group_attrs, array_attrs,
):
    group = zarr.group(overwrite=True,
                       store=zarr.N5Store(dest_path),
                       path=data_path)
    group.attrs.update(group_attrs)
    for n, s, d, c, a in zip(names, shapes, dtypes, chunks, array_attrs):
        g = group.zeros(name=n, shape=s, dtype=d, compressor=compressor, chunks=c)
        g.attrs.update(a)


def set_contrast_limits(path):
    arr = da.from_array(read(path))
    clims = da.compute([arr.min(), arr.max()])
    arr.attrs['contrast_limits'] = {'min': clims[0], 'max': clims[1]}
    return 0


def save_blockwise(v, store_path, split_dim, multiscale_level, block_info):
    # assumes input is 5D with first dimension == 1 and
    # second dimension == 2, and third + fourth dimensions the full size
    # of the zarr/n5 file located at store_path

    num_sinks = v.shape[split_dim]
    sinks = [
        zarr.open(store_path, mode="a")[f"/volumes/raw/ch{d}/s{multiscale_level}"] for d in range(num_sinks)
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

    parser.add_argument(
        "-ml",
        "--multiscale_levels",
        help="The number of multiscale levels to create, in addition to full resolution. E.g., if ml=4 (the default), "
             "downscaled data will be saved with downscaling factors of 2, 4, 8, 16, in addition to full-resolution.",
        default=4
    )

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, filemode="a", level=logging.INFO)
    num_workers = int(args.num_workers)
    multiscale_levels = range(0, args.multiscale_levels + 1)

    output_fmt = Path(args.dest).suffix[1:]
    if output_fmt not in OUTPUT_FMTS:
        raise NotImplementedError(
            f"Cannot write a chunked store using format {output_fmt}. Try one of {OUTPUT_FMTS}"
        )

    pyramid, metadata = prepare_data(args.source)
    pyramid = pyramid[:multiscale_levels[-1]]

    num_channels = pyramid[0].array.shape[channel_dim]

    dataset_paths = [f"/volumes/raw/ch{ch}" for ch in range(num_channels)]
    logging.info('Initializing chunked storage for multiple pyramid levels....')
    for dp in dataset_paths:
        chunks = ((1, max_chunksize, max_chunksize),) * len(pyramid)
        shapes, names, dtypes, array_attrs = [], [], [], []
        group_attrs = {'metadata': metadata}
        for ind, level in enumerate(pyramid):
            shapes.append(level.array.sum(channel_dim).shape)
            names.append(f's{ind}')
            dtypes.append(level.array.dtype)
            # resolution should go in here but we can't quite get that from the metadata yet
            _scale = list(level.scale_factors)
            _scale.pop(channel_dim)
            # zyx to xyz
            _scale = _scale[::-1]
            array_attrs.append({'downsamplingFactors':_scale})

        prepare_chunked_store(
            dest_path=args.dest,
            data_path=dp,
            names=names,
            shapes=shapes,
            dtypes=dtypes,
            compressor=compressor,
            chunks=chunks,
            group_attrs=group_attrs,
            array_attrs=array_attrs,
        )

    if not args.dry_run:
        running_on_cluster = bsub_available()
        if running_on_cluster:
            from fst.distributed import get_jobqueue_cluster

            cluster = get_jobqueue_cluster(project="cosem")
            cluster.scale(num_workers)
        else:
            from distributed import LocalCluster
            cluster = LocalCluster(n_workers=num_workers)

        client = Client(cluster)
        logging.info(
            f"Begin saving data to {args.dest}. View status at the following address: {cluster.dashboard_link}"
        )
        pyramid_saver = tuple(l.array.map_blocks(save_blockwise,
                                      store_path=args.dest,
                                      split_dim=channel_dim,
                                      dtype=pyramid[0].array.dtype,
                                      multiscale_level=ind) for ind, l in enumerate(pyramid))

        client.compute(pyramid_saver, sync=True)

        # logging.info(f"Calculating minimum and maximum intensity value of the data...")
        # [set_contrast_limits(args.dest + dataset_path + n) for n in n5_names]

        client.close()
        cluster.close()
        
        # zarr saves data in temporary files, which have very
        # restricted permissions. This function call recursively applies
        # new permissions to all the files  in the newly created container based on the current umask setting
        logging.info(f'Updating permissions of files in {args.dest}')
        chmodr(args.dest, mode='umask')
        
        elapsed_time = time.time() - start_time
        logging.info(f"Save completed in {elapsed_time} s")
