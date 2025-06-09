"""This module prepares ohlcv_resampler function."""
import pandas as pd
import numpy as np
from pandasohlcv import ohlcv_grouper  # pylint: disable=W0406

__slots__ = ("ohlcv_resampler",)


def ohlcv_resampler(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resamples dataframe to given frequency.

    Args:
        df (pd.DataFrame): Input dataframe.
        freq (str): Resampling frequency.

    Returns:
        pd.DataFrame: Resampled dataframe.
    """

    gpr = df.groupby(pd.Grouper(freq=freq)).grouper
    labels, _, ngroups = gpr.group_info
    out = np.empty((ngroups, 5), dtype=np.float64)
    out = np.ascontiguousarray(out)
    ohlcv_grouper.ohlcv_grouper(
        out, np.ascontiguousarray(df.values), np.ascontiguousarray(labels)
    )

    return pd.DataFrame(out, gpr.result_index, columns=df.columns)


def ohlcv_resampler_on_windows(windows: np.ndarray) -> np.ndarray:
    """Resamples OHLCV data from a 3D array of windows.

    This function takes a 3D NumPy array representing multiple windows of OHLCV (Open, High, Low, Close, Volume) data 
    and resamples it into a 2D array where each row corresponds to a window and contains the aggregated OHLCV values.

    Args:
        windows (np.ndarray): A 3D NumPy array of shape (n_windows, n_timesteps, 5) where:
            - n_windows: The number of windows.
            - n_timesteps: The number of time steps in each window.
            - 5: Represents the OHLCV components (Open, High, Low, Close, Volume).

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_windows, 5) containing the resampled OHLCV values for each window, 
        where each row corresponds to a window and the columns represent the aggregated Open, High, Low, Close, 
        and Volume values respectively.
    """
    assert windows.ndim == 3

    out = np.empty((windows.shape[0], 5), dtype=np.float64)
    out = np.ascontiguousarray(out)
    ohlcv_grouper.ohlcv_grouper_on_windows(out, np.ascontiguousarray(windows.copy()))

    return out
