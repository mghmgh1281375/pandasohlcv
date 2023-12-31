[![Python application](https://github.com/mghmgh1281375/pandasohlcv/actions/workflows/python-app.yml/badge.svg)](https://github.com/mghmgh1281375/pandasohlcv/actions/workflows/python-app.yml)
[![Pylint](https://github.com/mghmgh1281375/pandasohlcv/actions/workflows/pylint.yml/badge.svg)](https://github.com/mghmgh1281375/pandasohlcv/actions/workflows/pylint.yml)
![GitHub License](https://img.shields.io/github/license/mghmgh1281375/pandasohlcv)
![GitHub release (with filter)](https://img.shields.io/github/v/release/mghmgh1281375/pandasohlcv)



# Installation

`pip install https://github.com/mghmgh1281375/pandasohlcv/archive/refs/tags/v0.2.0.zip`


# Example
```python
from pandasohlcv import ohlcv_resampler
import time

t0 = time.time()
my_result = ohlcv_resampler(self.df, '5T')
print(round(time.time() - t0, 3))
```
