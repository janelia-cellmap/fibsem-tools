import argparse
from glob import glob
from pathlib import Path
from fst.io.ingest import padstack, arrays_from_delayed
from fst.io import read, access, rmtree_parallel, same_array_props
import numpy as np
import logging
import numcodecs
from fst.distributed import bsub_available, get_jobqueue_cluster
from distributed import Client, LocalCluster
import time
import dask.array as da
from fst.pyramid import lazy_pyramid
import zarr

OUTPUT_FMTS = {"n5"}
# the name of this program
program_name = 'dat_to_n5.py'

max_chunksize = 1024
compressor = numcodecs.GZip(level=-1)
# raw data are stored z c y x, we will split images into two channels along the channel dimension
channel_dim = 1
downscale_factor = 2

logger = logging.getLogger(program_name)
c_handler = logging.StreamHandler()
c_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_formatter)
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)


def prepare_data(path):
    if isinstance(path, str):
        fnames = sorted(glob(path))
    elif isinstance(path, list):
        fnames = path
    else:
        raise ValueError(f"Path variable should be string or list, not {type(path)}")
    # set the function to use for pyramid downscaling

    logger.info(f"Preparing {len(fnames)} images...")
    data_eager = read(fnames, lazy=False)
    meta, shapes, dtypes = [], [], []
    # build lists of metadata for each image
    for d in data_eager:
        meta.append(d.header.__dict__)
        shapes.append(d.shape)
        dtypes.append(d.dtype)

    data = arrays_from_delayed(read(fnames, lazy=True), shapes=shapes, dtypes=dtypes)
    # make sure these are numpy arrays, until we fix the issue with FIBSEMData returning a subclassed numpy array
    data = [d.map_blocks(np.array) for d in data]
    return data, meta


def prepare_pyramids(data, fill_values, chunks, reduction, downscale_factor):
    stacked, padding = padstack(data, constant_values=fill_values)
    stacked = stacked.rechunk(chunks)

    scaling_factors = (1, 1, downscale_factor, downscale_factor)
    pyramid = lazy_pyramid(stacked, reduction, scaling_factors, preserve_dtype=True)
    return pyramid, padding


def prepare_chunked_store(
        dest_path, data_path, names, shapes, dtypes, compressor, chunks, group_attrs, array_attrs,
):
    group = access(dest_path + data_path, mode='a')
    if isinstance(group, zarr.hierarchy.Array):
        rmtree_parallel(dest_path + data_path)
        group = access(dest_path + data_path, mode='a')

    group.attrs.clear()
    group.attrs.update(group_attrs)
    for n, s, d, c, a in zip(names, shapes, dtypes, chunks, array_attrs):
        # overwriting an array can take a long time, so we preempt that by manually deleting using dask-delayed
        try:
            arr = group.zeros(name=n, shape=s, dtype=d, compressor=compressor, chunks=c)
        except ValueError:
            arr = group[n]
            if not same_array_props(arr, shape=s, dtype=d, compressor=compressor, chunks=c):
                logger.info(f'Removing existing array at {dest_path + data_path + n}')
                rmtree_parallel(str(Path.joinpath(Path(dest_path + data_path), Path(n))))
                arr = group.zeros(name=n, shape=s, dtype=d, compressor=compressor, chunks=c)

        arr.attrs.clear()
        arr.attrs.update(a)


def set_contrast_limits(path):
    arr = access(path, mode='a')
    dar = da.from_array(arr)
    clims = da.compute(dar.min(), dar.max())
    arr.attrs['contrastLimits'] = {'min': int(clims[0]), 'max': int(clims[1])}
    return 0


def split_by_chunks(a):
    for i, sl in zip(np.ndindex(a.numblocks), da.core.slices_from_chunks(a.chunks)):
        yield (sl, a.blocks[i])


def save_blockwise(arr, path, block_info):
    sink = access(path, mode="a")
    pos = block_info[0]["array-location"]
    idx = tuple(slice(*i) for i in pos)
    sink[idx] = arr

    return np.zeros((1,) * arr.ndim)


def get_contrast_limits(data):
    results = []
    for fun in (da.min, da.max):
        results.append(fun(da.stack(fun(d, axis=(1, 2)) for d in data), axis=0))
    return da.stack(results)


def prepare_cluster():
    on_cluster = bsub_available()
    if on_cluster:
        cluster = get_jobqueue_cluster(project="cosem")
    else:
        cluster = LocalCluster()

    client = Client(cluster)
    logger.info(
        f"Begin saving data to {args.dest}. View status at the following address: {cluster.dashboard_link}"
    )
    return client


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
        "-ml",
        "--multiscale_levels",
        help="The number of multiscale levels to create, in addition to full resolution. E.g., if ml=6 (the default), "
             "downscaled data will be saved with downscaling factors of 2, 4, 8, 16, 32 in addition to full-resolution.",
        default=6
    )

    args = parser.parse_args()

    num_workers = int(args.num_workers)
    multiscale_levels = range(0, args.multiscale_levels + 1)
    chunks = (10, 2, -1, -1)
    output_fmt = Path(args.dest).suffix[1:]
    if output_fmt not in OUTPUT_FMTS:
        raise NotImplementedError(
            f"Cannot write a chunked store using format {output_fmt}. Try one of {OUTPUT_FMTS}"
        )

    client = prepare_cluster()
    data, metadata = prepare_data(args.source)
    client.cluster.scale(num_workers)
    logger.info('Calculating minimum and maximum values of the input data...')
    contrast_limits = get_contrast_limits(data).compute().T
    client.cluster.scale(0)

    pyramid, padding = prepare_pyramids(data, fill_values=contrast_limits.min(), chunks=chunks, reduction=np.mean, downscale_factor=downscale_factor)
    pyramid = pyramid[:multiscale_levels[-1]]
    logger.info(
        f"Assembled dataset with shape {pyramid[0].array.shape} using chunk sizes {pyramid[0].array.chunksize}"
    )
    scales = []
    for l in pyramid:
        _scale = list(l.scale_factors)
        _scale.pop(channel_dim)
        scales.append(_scale[::-1])

    num_channels = pyramid[0].array.shape[channel_dim]

    rawGroup = access(args.dest + '/volumes/raw', mode='a')
    rawGroup.attrs.clear()
    rawGroup.attrs['metadata'] = metadata

    dataset_paths = [f"/{rawGroup.path}/ch{ch}" for ch in range(num_channels)]

    logger.info('Initializing chunked storage for multiple pyramid levels....')
    for ind_d, dp in enumerate(dataset_paths):
        chunks = ((1, max_chunksize, max_chunksize),) * len(pyramid)
        shapes, names, dtypes, array_attrs = [], [], [], []
        group_attrs = {'downsamplingFactors': scales,
                       'padding': padding,
                       'createdBy': program_name,
                       'contrastLimits': contrast_limits[ind_d].tolist()}
        for ind_l, level in enumerate(pyramid):
            shapes.append(level.array.sum(channel_dim).shape)
            names.append(f's{ind_l}')
            dtypes.append(level.array.dtype)
            # resolution should go in here but we can't quite get that from the metadata yet

            array_attrs.append({'downsamplingFactors': scales[ind_l]})

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

    logger.info('Begin saving data...')
    to_store = []
    for ind_l, l in enumerate(pyramid):
        for ind_c in range(l.array.shape[channel_dim]):
            path = f'{args.dest}/volumes/raw/ch{ind_c}/s{ind_l}'
            store = da.take(l.array, ind_c, axis=channel_dim).map_blocks(save_blockwise, path, dtype='int')
            to_store.append(store)

    client.cluster.scale(num_workers)
    client.compute(to_store, sync=True)
    client.cluster.scale(0)

    elapsed_time = time.time() - start_time
    logger.info(f"Save completed in {elapsed_time} s")
