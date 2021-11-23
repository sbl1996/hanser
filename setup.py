#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
import glob

from setuptools import find_packages, setup, Extension

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

# Package meta-data.

NAME = 'hanser'
IMPORT_NAME = 'hanser'
DESCRIPTION = "HrvvI's extension to TensorFlow"
URL = 'https://github.com/sbl1996/hanser'
EMAIL = 'sbl1996@126.com'
AUTHOR = 'HrvvI'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

REQUIRED = [
    "Pillow",
    "numpy",
    "toolz",
    "pybind11",
    "cerberus",
    "tensorflow_probability==0.11.1",
    "tensorflow_datasets>=4.3.0",
    "hhutil",
    "lark",
    "pendulum",
    "loguru",
    "typer",
]

DEPENDENCY_LINKS = [
]

here = os.path.dirname(os.path.abspath(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's _version.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, IMPORT_NAME, '_version.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


def get_pybind_include(user=False):
    import pybind11
    return pybind11.get_include(user)


def get_numpy_extensions():
    extensions_dir = os.path.join(here, IMPORT_NAME, 'csrc', 'numpy')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))

    extra_compile_args = []
    if sys.platform == 'darwin':
        extra_compile_args += ['-stdlib=libc++',
                               '-mmacosx-version-min=10.9']

    include_dirs = [
        extensions_dir,
        get_pybind_include(),
        get_pybind_include(user=True),
    ]

    ext_modules = [
        Extension(
            IMPORT_NAME + '._numpy',
            main_file,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    dependency_links=DEPENDENCY_LINKS,
    # include_package_data=True,
    license='MIT',
    # ext_modules=get_numpy_extensions(),
)
