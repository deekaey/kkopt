# -*- coding: utf-8 -*-
import os
import glob
from setuptools import setup, find_packages

DISTNAME = "kkopt"
LICENSE = "MIT"
AUTHOR = "David Kraus, Steffen Klatt"
AUTHOR_EMAIL = "david.kraus@gmx.de"
URL = "https://github.com/deekaey/kkopt"
DESCRIPTION = "..."
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering",
]
PYTHON_REQUIRES = ">=3.7"

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

INSTALL_REQUIRES = [
    "python-dotenv",  # if 'dotenv' really means this package name
    "numexpr",
    "spotpy",
    "numpy",
    "pandas",
    "pyyaml",
    "seaborn",
    "mpi4py",
    "SALib",          # correct PyPI name
    "kkplot @ git+https://github.com/deekaey/kkplot.git",
]

setup(
    name=DISTNAME,
    version="0.0.1",
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    python_requires=PYTHON_REQUIRES,
    packages=find_packages("src", exclude=["docs", "tests"]),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob("src/*.py")
    ],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "kkopt=kkopt.kkopt:main",
        ]
    },
)
