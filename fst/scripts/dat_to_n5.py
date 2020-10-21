import argparse
from glob import glob
from pathlib import Path
from fst.io.ingest import pad_arrays, arrays_from_delayed
from fst.io import read, access, rmtree_parallel, create_arrays
import numpy as np
import os
import logging
import numcodecs
from fst.distributed import bsub_available, get_jobqueue_cluster, get_cluster
from toolz import partition_all
from typing import Tuple, List, Union

import time
import dask
import dask.array as da
from fst.pyramid import lazy_pyramid, get_downsampled_offset
import zarr

OUTPUT_DTYPES = ("same", "uint8", "uint16")
OUTPUT_FMTS = ("n5",)

program_name = "dat_to_n5.py"

grid_spacing_unit = "nm"
max_chunksize = 1024
compressor = numcodecs.GZip(level=-1)
# all channels will the stored as subgroups under root_group_path
root_group_path = Path("volumes/raw/")
# raw data are stored z | c y x, we will split images into two channels along the channel axis
channel_dim = 0
downscale_factor = 2

# set up logging
logger = logging.getLogger(program_name)
c_handler = logging.StreamHandler()
c_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_formatter)
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)


def prepare_data(path: Union[str, list]) -> Tuple[list, list]:
    """
    Takes a path to a collection of .dat files, specified either as single path or a list of individual files,
    and generates list of dask arrays representing the dataset.
    """
    if isinstance(path, str):
        fnames = sorted(glob(path + "*.dat"))
    elif isinstance(path, list):
        fnames = sorted(path)
    else:
        raise ValueError(f"Path variable should be string or list, not {type(path)}")

    logger.info(f"Preparing {len(fnames)} images...")
    data_eager = read(fnames, lazy=False)
    meta, shapes, dtypes = [], [], []
    # build lists of metadata for each image
    for d in data_eager:
        meta.append(d.header.__dict__)
        shapes.append(d.shape)
        dtypes.append(d.dtype)

    data = arrays_from_delayed(read(fnames, lazy=True), shapes=shapes, dtypes=dtypes)
    return data, meta


def read_grid_spacing(metadata: list) -> dict:
    """
    Report the grid spacing of the data in nanometers, given the metadata for an entire
    FIBSEM dataset. The lateral resolution can be inferred from the `PixelSize` attribute of a single
    image, while the axial resolution is an estimate based on the total change in working distance of the SEM
    beam divided by the total number of slices.
    """
    lateral = metadata[0]["PixelSize"]
    axial = abs((metadata[0]["WD"] - metadata[-1]["WD"]) / len(metadata)) / 1e-6
    return {"z": axial, "y": lateral, "x": lateral}


def read_minmax(array_paths: Union[List[str], List[Path]]) -> np.array:
    minmax = np.zeros((len(array_paths), 2))
    for ind, pth in enumerate(array_paths):
        lims = np.array([np.NaN, np.NaN])
        try:
            lims = read(pth).attrs.get("rawHistogramBounds", np.array([np.NaN, np.NaN]))
        except ValueError:
            pass
        minmax[ind] = lims
    return minmax


def calc_minmax(data: list) -> list:
    @dask.delayed
    def minmax(v):
        return np.array((v.min((1, 2)), v.max((1, 2))))

    return list(map(minmax, data))


def prepare_pyramids(
    data, fill_values, chunks, reduction, downscale_factor, output_dtype
) -> Tuple[list, list]:
    padded, padding = pad_arrays(data, constant_values=fill_values)
    stacked = da.stack(padded).rechunk(chunks)
    if output_dtype != "same":
        stacked = (stacked - fill_values.reshape(1, -1, 1, 1)).astype(output_dtype)
    scaling_factors = (1, 1, downscale_factor, downscale_factor)
    pyramid = lazy_pyramid(stacked, reduction, scaling_factors, preserve_dtype=True)
    return pyramid, padding


def change_dtype(data: list, output_dtype: str, offset: np.array) -> list:
    """
    Lazy histogram-preserving datatype adjustment of a collection of array-likes. 
    Signed datatypes (int8, int16) are converted to their unsigned counterparts (uint8, uint16) by upcasting to signed type with 
    higher precision, shifting all values by a constant, then downcasting to the final unsigned datatype. The resulting arrays 
    have a global minimum of 0, with the original min-max distance.  
    """

    @dask.delayed
    def adjuster(arr, upcast, offset, dtype):
        assert arr.ndim == offset.ndim
        return (arr.astype(upcast) - offset).astype(dtype)

    if output_dtype == "same":
        return data
    elif output_dtype == "uint8":
        assert (data[0].dtype == "int8") or (data[0].dtype == ">i1")
        upcast = "int16"
    elif output_dtype == "uint16":
        assert (data[0].dtype == "int16") or (data[0].dtype == ">i2")
        upcast = "int32"

    return [
        da.from_delayed(
            adjuster(
                d, upcast=upcast, offset=offset.reshape(-1, 1, 1), dtype=output_dtype
            ),
            dtype=output_dtype,
            shape=d.shape,
        )
        for d in data
    ]


@dask.delayed
def save_pyramid_slicewise(arr, z_index, scale_factors, reduction, depth, output_path):
    pyramid = lazy_pyramid(arr, reduction, scale_factors)[:depth]
    for ind_l, pyr in enumerate(pyramid):
        layer = pyr.data.compute(scheduler="synchronous")
        for ind_ch, ch in enumerate(layer):
            dataset_path = Path(output_path) / f"ch{ind_ch}" / f"s{ind_l}"
            sink = access(dataset_path, mode="a")
            sink[z_index] = ch
    return 0


def dat_to_n5(
    source: str,
    dest: str,
    num_workers: Union[str, int],
    multiscale_levels: int,
    datatype: str,
    output_chunks: tuple,
):
    start_time = time.time()
    num_workers = int(num_workers)
    levels = np.arange(0, multiscale_levels + 1)
    output_fmt = Path(dest).suffix[1:]

    if output_fmt not in OUTPUT_FMTS:
        raise NotImplementedError(
            f"Cannot write a chunked store using format {output_fmt}. Try one of {OUTPUT_FMTS}"
        )

    if datatype not in OUTPUT_DTYPES:
        raise ValueError(f"Datatype must be one of {OUTPUT_DTYPES}")

    client = get_cluster(threads_per_worker=1)
    logger.info(f"Observe progress at {client.cluster.dashboard_link}")

    data, metadata = prepare_data(source)
    grid_spacing = read_grid_spacing(metadata)
    grid_spacing_n5 = (grid_spacing["x"], grid_spacing["y"], grid_spacing["z"])
    num_channels = data[0].shape[channel_dim]

    root_group = access(Path(dest) / root_group_path, mode="a")
    root_group.attrs.put({"metadata": metadata})
    dataset_paths = [
        Path(dest) / root_group_path / f"ch{ch}" for ch in range(num_channels)
    ]

    minmax = read_minmax(dataset_paths)

    if np.any(np.isnan(minmax)):
        client.cluster.scale(num_workers)
        logger.info("Calculating minimum and maximum values of the input data...")
        minmax_per_image = np.array(client.compute(calc_minmax(data), sync=True))
        client.cluster.scale(0)
        minmax = np.array((minmax_per_image.min((0, 1)), minmax_per_image.max((0, 1))))
    else:
        logger.info(
            "Using previously calculated minimum and maximum values of the input data..."
        )

    data_adjusted = change_dtype(data, output_dtype=datatype, offset=minmax[0])
    padded, padding = pad_arrays(data_adjusted, constant_values=0)
    logger.info(f"Assembled dataset with shape {(len(padded), *padded[0].shape)}")

    sample_pyramid = lazy_pyramid(
        np.take(padded[0], 0, channel_dim),
        np.mean,
        (downscale_factor, downscale_factor),
    )[: levels[-1]]

    scales = []
    for l in sample_pyramid:
        scale = list(l.scale_factors)
        scales.append([*scale, 1])

    logger.info("Initializing storage for multiple pyramid levels....")
    for ind_d, dp in enumerate(dataset_paths):
        chunks = [output_chunks] * len(padded)
        shapes, names, dtypes, array_attrs = [], [], [], []
        group_attrs = {
            "downsamplingFactors": scales,
            "padding": padding,
            "createdBy": program_name,
            "rawHistogramBounds": minmax[ind_d].tolist(),
            "datatypeChange": datatype != "same",
        }
        for ind_l, level in enumerate(sample_pyramid):
            shapes.append([len(padded), *level.shape])
            names.append(f"s{ind_l}")
            dtypes.append(level.dtype)
            array_attrs.append(
                {
                    "downsamplingFactors": scales[ind_l],
                    "pixelResolution": {
                        "dimensions": np.multiply(
                            grid_spacing_n5, scales[ind_l]
                        ).tolist(),
                        "units": grid_spacing_unit,
                    },
                    "offset": get_downsampled_offset(scales[ind_l]).tolist(),
                }
            )
        logger.info(f"Initializing {dp}...")
        create_arrays(
            path=dp,
            names=names,
            shapes=shapes,
            dtypes=dtypes,
            compressors=[compressor for n in names],
            chunks=chunks,
            group_attrs=group_attrs,
            array_attrs=array_attrs,
        )

    logger.info("Begin saving data...")

    to_store = [
        save_pyramid_slicewise(
            arr=sl,
            z_index=z_idx,
            scale_factors=(1, downscale_factor, downscale_factor),
            reduction=np.mean,
            depth=levels[-1],
            output_path=f"{dest}/{root_group_path}",
        )
        for (z_idx, sl) in enumerate(padded)
    ]

    partition_size = 10000
    if len(to_store) < partition_size:
        partition_size = len(to_store)

    partitions = partition_all(partition_size, to_store)

    for ind_p, p in enumerate(partitions):
        client.cluster.scale(num_workers)
        logger.info(
            f"Begin saving planes {partition_size * ind_p}:{partition_size * ind_p + len(p)}"
        )
        client.compute(p, sync=True)
        client.cluster.scale(0)

    elapsed_time = time.time() - start_time
    logger.info(f"Save completed in {elapsed_time} s")
    return 0


def main():
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
        default=6,
    )

    parser.add_argument(
        "-dt",
        "--datatype",
        help="The datatype to use for the output. Supported datatypes are `same`, "
        '`uint8`, and `uint16`. Default value is "same", which preserves the '
        "original datatype",
        default="same",
    )

    args = parser.parse_args()
    return dat_to_n5(
        source=args.source,
        dest=args.dest,
        num_workers=args.num_workers,
        multiscale_levels=args.multiscale_levels,
        datatype=args.datatype,
        output_chunks=(1, max_chunksize, max_chunksize),
    )


if __name__ == "__main__":
    main()
