[![Python application](https://github.com/mghmgh1281375/pandasohlcv/actions/workflows/python-app.yml/badge.svg)](https://github.com/mghmgh1281375/pandasohlcv/actions/workflows/python-app.yml)

# Installation

1. `git clone https://github.com/mghmgh1281375/pandasohlcv.git`
2. `cd pandasohlcv`
3. `python setup.py build_ext --inplace`

# Example
```python
from pandasohlcv import ohlcv_resampler
import time

t0 = time.time()
my_result = ohlcv_resampler(self.df, '5T')
print(round(time.time() - t0, 3))
```
