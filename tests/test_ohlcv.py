"""
Module Docstring

This module contains a test case class 'OHLCVTestCase' for testing the OHLCV
    (Open, High, Low, Close, Volume) functionality.
The test case class is designed to set up a common test environment and
    includes test methods for both normal and NaN scenarios.

The module also defines a static method 'asscalar' for converting a NumPy
    array with a single element to a scalar.
Additionally, the test case class includes two test methods: 'test_ohlcv' and
    'test_ohlcv_nan', each evaluating
different scenarios of the 'ohlcv_resampler' function.

- Author: Mohammad Ghorbani
- Version: 0.1.0
- Dependencies: numpy, pandas, pandasohlcv

"""

import time
import unittest
import numpy as np
import pandas as pd
from pandasohlcv import ohlcv_resampler, ohlcv_resampler_on_windows


class OHLCVTestCase(unittest.TestCase):
    """
    Test case class for the OHLCV functionality.

    This class sets up a common test environment for testing OHLCV (Open, High
        , Low, Close, Volume) data operations.

    Attributes:
    - cond (dict): A dictionary defining aggregation conditions for each OHLCV
        column.
    - df (pandas.DataFrame): A DataFrame with randomly generated OHLCV data
        for testing.

    Methods:
    - __init__(methodName: str = "runTest"): Constructor method to initialize
        the test case.
    """

    def __init__(self, methodName: str = "runTest") -> None:
        """
        Initialize the OHLCVTestCase.

        Parameters:
        - methodName (str): The name of the test method to run. Default is
            "runTest".
        """
        super().__init__(methodName)
        np.random.seed(42)
        self.cond = {
            "open": lambda a: self.asscalar(a.dropna().first("T").values),
            "high": np.nanmax,
            "low": np.nanmin,
            "close": lambda a: self.asscalar(a.dropna().last("T").values),
            "volume": lambda a: np.nansum(a) if not np.isnan(a).all() else np.nan,
        }

        columns = ["open", "high", "low", "close", "volume"]
        index = pd.date_range("2023-01-01", "2023-02-02", freq="T")
        arr = np.random.randint(0, 10, size=(len(index), 5))
        arr = arr.astype(np.float64)
        self.df = pd.DataFrame(arr, index=index, columns=columns)

    @staticmethod
    def asscalar(arr):
        """
        Convert a NumPy array with a single element to a scalar.

        Parameters:
        - arr (numpy.ndarray): Input NumPy array.

        Returns:
        - float or np.nan: The scalar value extracted from the input array.
            If the array is empty, returns np.nan.

        Example:
        ```
        import numpy as np
        result = OHLCVTestCase.asscalar(np.array([42.0]))
        print(result)  # Output: 42.0
        ```
        """

        if len(arr) > 0:
            return np.float64(arr.item())
        return np.nan

    def test_ohlcv(self):
        """
        Test the 'ohlcv_resampler' function without NaN values.

        This test function generates a DataFrame,
            applies the 'ohlcv_resampler' function,
        and compares the result with the equivalent Pandas resampling
            operation.

        Test Steps:
        1. Create a copy of the original DataFrame.
        2. Measure the time taken by 'ohlcv_resampler' and Pandas resampling.
        3. Compare the results obtained from 'ohlcv_resampler' and Pandas
            resampling.
        4. Assert that values in the results are equal.
        """
        t0 = time.time()
        my_result = ohlcv_resampler(self.df, "5T")
        print(round(time.time() - t0, 3))

        t0 = time.time()
        pd_result = self.df.resample("5T").agg(self.cond)
        print(round(time.time() - t0, 3))

        np.testing.assert_array_equal(my_result.values, pd_result.values)

    def test_ohlcv_nan(self):
        """
        Test the 'ohlcv_resampler' function with NaN values.

        This test function generates a DataFrame with random NaN values,
            applies the 'ohlcv_resampler' function,
        and compares the result with the equivalent Pandas resampling
            operation. The NaN values are introduced
        into the DataFrame to simulate a real-world scenario.

        Test Steps:
        1. Create a copy of the original DataFrame.
        2. Introduce random NaN values to the DataFrame using a mask.
        3. Measure the time taken by 'ohlcv_resampler' and Pandas resampling.
        4. Compare the results obtained from 'ohlcv_resampler' and Pandas
            resampling.
        5. Assert that non-NaN values in the results are equal.
        """
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

    @staticmethod
    def ohlcv_aggregation(ohlcv: np.ndarray) -> np.ndarray:
        def _reshape(_ohlcv):
            for _g in _ohlcv:
                yield _g.reshape(-1, 5)

        return np.array(
            [
                [
                    g[(np.isfinite(g[:, 0])).argmax(), 0],
                    np.nanmax(g[:, 1]),
                    np.nanmin(g[:, 2]),
                    g[g.shape[0] - (np.isfinite(g[::-1, 3])).argmax() - 1, 3],
                    np.nansum(g[:, 4]),
                ]
                for g in _reshape(ohlcv)
            ]
        )

    def test_ohlcv_on_windows(self):
        """
        Test the ohlcv_resampler_on_windows function against a pandas-based
            aggregation method.

        This test function generates a random time series DataFrame with OHLCV
            data and compares the results of the Cython-based
            ohlcv_resampler_on_windows function
        with a pandas-based aggregation method.

        Parameters:
        None

        Returns:
        None

        Raises:
        AssertionError: If the results of the Cython-based and pandas-based
            methods differ.

        Example:
        ```
        test_instance = YourTestClass()
        test_instance.test_ohlcv_on_windows()
        ```

        Note:
        This function compares the results of Cython-based and pandas-based
            methods for resampling OHLCV (open, high, low, close, volume) data
            on windows.
        The test involves creating a random DataFrame, applying resampling,
            and comparing the results.
        """

        columns = ["open", "high", "low", "close", "volume"]
        index = pd.date_range("2023-01-01", "2024-02-02", freq="T")
        arr = np.random.randint(0, 10, size=(len(index), 5))
        arr = arr.astype(np.float64)
        df = pd.DataFrame(arr, index=index, columns=columns)

        nan_mask = np.random.random(df.shape) > 0.8
        df.values[nan_mask] = np.nan

        t0 = time.time()
        df = ohlcv_resampler(df, "T")
        print(round(time.time() - t0, 3))

        t0 = time.time()
        n = 3
        my_df_list = []
        for i in range(0, n):
            shifted = df.iloc[i:]
            arr = np.pad(
                np.array(shifted), [[0, n - 1], [0, 0]], constant_values=np.nan
            )
            windows = np.lib.stride_tricks.sliding_window_view(arr, (n, 5))[::n]
            aggregated = ohlcv_resampler_on_windows(windows.copy().squeeze())
            aggregated = pd.DataFrame(
                aggregated, index=shifted[::n].index, columns=df.columns
            )
            my_df_list.append(aggregated)

        print(round(time.time() - t0, 3), "cython")

        t0 = time.time()
        n = 3
        pd_df_list = []
        for i in range(0, n):
            shifted = df.iloc[i:]
            arr = np.pad(
                np.array(shifted), [[0, n - 1], [0, 0]], constant_values=np.nan
            )
            windows = np.lib.stride_tricks.sliding_window_view(arr, (n, 5))[::n]
            aggregated = self.ohlcv_aggregation(windows)
            aggregated = pd.DataFrame(
                aggregated, index=shifted[::n].index, columns=df.columns
            )
            pd_df_list.append(aggregated)

        print(round(time.time() - t0, 3), "ohlcv_aggregate")

        for i in range(len(my_df_list)):
            my_df_list[i].loc[:, "volume"] = my_df_list[i].loc[:, "volume"].fillna(0)
            my_nan_mask = np.isnan(my_df_list[i])
            pd_nan_mask = np.isnan(pd_df_list[i])

            np.testing.assert_array_equal(
                my_df_list[i].values[~my_nan_mask], pd_df_list[i].values[~pd_nan_mask]
            )
