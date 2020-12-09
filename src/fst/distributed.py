#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Boilerplate for setting up a dask.distributed environment on the janelia compute cluster.
#
# Davis Bennett
# davis.v.bennett@gmail.com
#
# License: MIT
#

from shutil import which
import dask

dask.config.set({"jobqueue.lsf.use-stdin": True})
dask.config.set({"distributed.comm.timeouts.connect": 60})
import dask.array as da
from distributed import Client, LocalCluster
from dask_jobqueue import LSFCluster
import os
import numpy as np
from pathlib import Path

# this is necessary to ensure that workers get the job script from stdin


def get_jobqueue_cluster(
    walltime="1:00",
    ncpus=1,
    cores=1,
    memory="16GB",
    threads_per_worker=1,
    death_timeout="600s",
    **kwargs,
):
    """
    Instantiate a dask_jobqueue cluster using the LSF scheduler on the Janelia Research Campus compute cluster.
    This function wraps the class dask_jobqueue.LSFCLuster and instantiates this class with some sensible defaults.
    Extra kwargs added to this function will be passed to LSFCluster().
    The full API for the LSFCluster object can be found here:
    https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster
    """

    if threads_per_worker == 1:
        env_extra = [
            "export NUM_MKL_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export OPENMP_NUM_THREADS=1",
            "export OMP_NUM_THREADS=1",
        ]
    else:
        raise ValueError("threads_per_worker can only be 1")

    USER = os.environ["USER"]
    HOME = os.environ["HOME"]

    if "local_directory" not in kwargs:
        kwargs["local_directory"] = f"/scratch/{USER}/"

    if "log_directory" not in kwargs:
        log_dir = f"{HOME}/.dask_distributed/"
        Path(log_dir).mkdir(parents=False, exist_ok=True)
        kwargs["log_directory"] = log_dir

    cluster = LSFCluster(
        walltime=walltime,
        cores=cores,
        ncpus=ncpus,
        memory=memory,
        env_extra=env_extra,
        death_timeout=death_timeout,
        **kwargs,
    )
    return cluster


def bsub_available() -> bool:
    """

    Returns True if the `bsub` command is available on the path, False otherwise. This is used to check whether code is
    running on the Janelia Compute Cluster.

    -------

    """
    result = which("bsub") is not None
    return result


def get_cluster(**kwargs):
    """
    Create a dask.distributed Client object with either a Jobqueue cluster (for use on the Janelia Compute Cluster)
    or a LocalCluster (for use on a single machine). Keyword arguments given to this function will be forwarded to
    the `get_jobqueue_cluster` function or the LocalCluster constructor.
    """
    if bsub_available():
        cluster = get_jobqueue_cluster(**kwargs)
    else:
        if "host" not in kwargs:
            kwargs["host"] = ""
        cluster = LocalCluster(**kwargs)

    client = Client(cluster, set_as_default=False)
    return client


def blockwise(arr):
    """
    A generator that yields (slice, block) tuples from a dask array. This effectively breaks a dask array into separate
    chunks.
    """
    for i, sl in zip(np.ndindex(arr.numblocks), da.core.slices_from_chunks(arr.chunks)):
        yield (sl, arr.blocks[i])
