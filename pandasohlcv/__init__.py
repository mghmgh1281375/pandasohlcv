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
