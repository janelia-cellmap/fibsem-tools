#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py file adapted from https://github.com/navdeep-G/setup.py
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = "fst"
DESCRIPTION = "Tools for processing FIB-SEM data."
URL = "https://github.com/janelia-cosem/fst"
EMAIL = "bennettd@janelia.hhmi.org"
AUTHOR = "Davis Bennett"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = "0.1.0"
REQUIRED = ["dask", "numpy", "zarr", "dask_jobqueue", "xarray", "h5py"]

EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
