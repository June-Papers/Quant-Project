from typing import Dict
from pathlib import Path
import pandas as pd


def build_market_benchmark(close: pd.DataFrame, cap: pd.DataFrame) -> Dict[str, pd.Series]:
    """Build a market benchmark return series from KOSPI-style market cap weights."""
    close_df = close.copy()
    cap_df = cap.copy()

    close_df["date"] = pd.to_datetime(close_df["date"]) if "date" in close_df.columns else close_df.index
    cap_df["date"] = pd.to_datetime(cap_df["date"]) if "date" in cap_df.columns else cap_df.index

    close_df = close_df.set_index("date").sort_index()
    cap_df = cap_df.set_index("date").sort_index()

    common_cols = sorted(set(close_df.columns) & set(cap_df.columns))
    close_df = close_df[common_cols]
    cap_df = cap_df[common_cols]

    returns = close_df.pct_change()
    weight = cap_df.div(cap_df.sum(axis=1), axis=0).shift(1).fillna(0)
    benchmark_ret = (weight * returns).sum(axis=1).fillna(0)
    benchmark_cum = (1 + benchmark_ret).cumprod()

    return {"return": benchmark_ret, "cum_return": benchmark_cum}
