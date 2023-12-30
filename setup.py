from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
 
setup(
    name='pandasohlcv',
    version='0.1.0',
    author="Mohammad Ghorbani",
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.4", "Cython==3.0.7", "pandas==1.3.4"
    ],
    python_requires=">=3.9.5, <4",
    ext_modules=cythonize("pandasohlcv/ohlcv_grouper.pyx"),
    include_dirs=[numpy.get_include()]
)
