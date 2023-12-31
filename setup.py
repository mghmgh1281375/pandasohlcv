"""
pandasohlcv Setup Script

This script is used to configure the installation of the pandasohlcv package
    using setuptools.

- Author: Mohammad Ghorbani
- Version: 0.2.0
- Python Version: >=3.9.5, <4

Dependencies:
The required dependencies for this package are listed in the 'requirements.txt' file.

Usage:
- To install the package and its dependencies, use: 'pip install .'
- To build the Cython extension module, 'ohlcv_grouper.pyx',
    use: 'python setup.py build_ext --inplace'

Note:
Make sure to have Cython and numpy installed before building the extension
    module.

"""

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="pandasohlcv",
    version="0.2.0",
    author="Mohammad Ghorbani",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9.5, <4",
    ext_modules=cythonize("pandasohlcv/ohlcv_grouper.pyx"),
    include_dirs=[numpy.get_include()],
)
