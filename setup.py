from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="pandasohlcv",
    version="0.1.0",
    author="Mohammad Ghorbani",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9.5, <4",
    ext_modules=cythonize("pandasohlcv/ohlcv_grouper.pyx"),
    include_dirs=[numpy.get_include()],
)
