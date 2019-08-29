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

from dask_jobqueue import LSFCluster
import os

def get_jobqueue_cluster(
    walltime="12:00",
    cores=1,
    local_directory=None,
    memory="16GB",
    env_extra=None,
    **kwargs
):
    """
    Instantiate a dask_jobqueue cluster using the LSF scheduler on the Janelia Research Campus compute cluster.
    This function wraps the class dask_jobqueue.LSFCLuster and instantiates this class with some sensible defaults.
    Extra kwargs added to this function will be passed to LSFCluster().
    The full API for the LSFCluster object can be found here:
    https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster
    """

    if env_extra is None:
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
        local_directory=local_directory,
        memory=memory,
        env_extra=env_extra,
        job_extra=['-o /dev/null'],
        **kwargs
    )
    return cluster