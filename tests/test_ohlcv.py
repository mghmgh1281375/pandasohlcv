import time
import unittest
import numpy as np
import pandas as pd
from pandasohlcv import ohlcv_resampler


class OHLCVTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        np.random.seed(42)
        self.cond = {
            'open': lambda a: self.asscalar(a.dropna().first("T").values),
            'high': np.nanmax,
            'low': np.nanmin,
            'close': lambda a: self.asscalar(a.dropna().last("T").values),
            'volume': lambda a: np.nansum(a) if not np.isnan(a).all() else np.nan,
        }

        columns = ["open", "high", "low", "close", "volume"]
        index = pd.date_range("2023-01-01", "2023-02-02", freq="T")
        arr = np.random.randint(0, 10, size=(len(index), 5))
        arr = arr.astype(np.float64)
        self.df = pd.DataFrame(arr, index=index, columns=columns)

    @staticmethod
    def asscalar(arr):
        if len(arr) > 0:
            return np.float64(arr.item())
        return np.nan

    def test_ohlcv(self):
        t0 = time.time()
        my_result = ohlcv_resampler(self.df, "5T")
        print(round(time.time() - t0, 3))

        t0 = time.time()
        pd_result = self.df.resample("5T").agg(self.cond)
        print(round(time.time() - t0, 3))

        np.testing.assert_array_equal(my_result.values, pd_result.values)

    def test_ohlcv_nan(self):
        df = self.df.copy()
        nan_mask = np.random.random(df.shape) > 0.8
        df.values[nan_mask] = np.nan

        t0 = time.time()
        my_result = ohlcv_resampler(df, "5T")
        print(round(time.time() - t0, 3))

        t0 = time.time()
        pd_result = df.resample("5T").agg(self.cond)
        print(round(time.time() - t0, 3))

        my_nan_mask = np.isnan(my_result.values)
        pd_nan_mask = np.isnan(pd_result.values)
        np.testing.assert_array_equal(
            my_result.values[~my_nan_mask], pd_result.values[~pd_nan_mask]
        )
