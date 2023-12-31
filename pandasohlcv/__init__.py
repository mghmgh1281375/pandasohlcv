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
    """_summary_

    Args:
        windows (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    assert windows.ndim == 3

    out = np.empty((windows.shape[0], 5), dtype=np.float64)
    out = np.ascontiguousarray(out)
    ohlcv_grouper.ohlcv_grouper_on_windows(out, np.ascontiguousarray(windows.copy()))

    return out
