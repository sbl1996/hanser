#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
import glob

from setuptools import find_packages, setup, Extension
import pkg_resources

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
    "tensorflow_datasets>=4.3.0",
    "hhutil",
    "lark",
    "pendulum",
    "loguru",
    "typer",
    'typeguard',
]

tfp_version_compat_table = {
    "2.7": "0.15.0",
    "2.6": "0.14.1",
    "2.5": "0.13.0",
    "2.4": "0.12.2",
    "2.3": "0.11.1",
}

def get_tf_version():
    try:
        version = pkg_resources.get_distribution("tensorflow").version
    except pkg_resources.DistributionNotFound:
        version = pkg_resources.get_distribution("tf_nightly").version
    return version.rsplit('.', 1)[0]

tfp_version = tfp_version_compat_table[get_tf_version()]
REQUIRED.append(f"tensorflow_probability=={tfp_version}")

here = os.path.dirname(os.path.abspath(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

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
    dependency_links=[],
    # include_package_data=True,
    license='MIT',
    # ext_modules=get_numpy_extensions(),
)
