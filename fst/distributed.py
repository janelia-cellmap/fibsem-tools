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

# todo: make this a class that inherits from dask_jobqueue.LSFCluster
from shutil import which
import dask
from dask_jobqueue import LSFCluster
import os

# this is necessary to ensure that workers get the job script from stdin
dask.config.set({"jobqueue.lsf.use-stdin": True})


def get_jobqueue_cluster(
    walltime="1:00",
    ncpus=1,
    cores=1,
    local_directory=None,
    memory="16GB",
    env_extra='single-threaded',
    **kwargs
):
    """
    Instantiate a dask_jobqueue cluster using the LSF scheduler on the Janelia Research Campus compute cluster.
    This function wraps the class dask_jobqueue.LSFCLuster and instantiates this class with some sensible defaults.
    Extra kwargs added to this function will be passed to LSFCluster().
    The full API for the LSFCluster object can be found here:
    https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster
    """

    if env_extra == 'single-threaded':
        env_extra = [
            "export NUM_MKL_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export OPENMP_NUM_THREADS=1",
            "export OMP_NUM_THREADS=1",
        ]

    if local_directory is None:
        local_directory = "/scratch/" + os.environ["USER"] + "/"

    cluster = LSFCluster(
        queue="normal",
        walltime=walltime,
        cores=cores,
        ncpus=ncpus,
        local_directory=local_directory,
        memory=memory,
        env_extra=env_extra,
        job_extra=["-o /dev/null"],
        **kwargs
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
