"""Universe filtering utilities for factor backtesting."""

from typing import Optional

import pandas as pd


def _prepare_daily_df(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    temp = df.copy()
    if date_col in temp.columns:
        temp[date_col] = pd.to_datetime(temp[date_col])
        temp = temp.set_index(date_col)
    elif isinstance(temp.index, pd.DatetimeIndex):
        temp.index = pd.to_datetime(temp.index)
    else:
        raise ValueError("Universe inputs must contain a 'date' column or a DatetimeIndex")
    return temp.sort_index()


def _rebalance_dates(index: pd.DatetimeIndex, reb_freq: str = "M") -> pd.DatetimeIndex:
    reb_freq = reb_freq.upper()
    if reb_freq == "D":
        return index
    if reb_freq in ("W", "M", "Q", "Y"):
        return (
            index.to_series()
            .groupby(index.to_period(reb_freq))
            .first()
            .values
        )
    raise ValueError("reb_freq must be 'D','W','M','Q' or 'Y'")


def _previous_month_end(date: pd.Timestamp) -> pd.Timestamp:
    return (date.to_period("M") - 1).to_timestamp("M")


def build_universe_mask(
    close: pd.DataFrame,
    cap: pd.DataFrame,
    volume: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
    halted: pd.DataFrame,
    cap_threshold: float = 100000.0,
    turnover_quantile: float = 0.05,
    exclude_halted: bool = True,
    reb_freq: str = "M"
) -> pd.DataFrame:
    """Build a universe inclusion mask for each rebalance date.

    Parameters
    ----------
    close : pd.DataFrame
        Daily close prices with a 'date' column or DatetimeIndex.
    cap : pd.DataFrame
        Daily market cap with the same format.
    volume : pd.DataFrame
        Daily volume.
    shares_outstanding : pd.DataFrame
        Daily shares outstanding.
    halted : pd.DataFrame
        Daily halted flag (1 or 0).
    cap_threshold : float
        Minimum market cap in 백만원. Default 100000 = 1000억원.
    turnover_quantile : float
        Lowest turnover quantile to exclude. Default 0.05.
    exclude_halted : bool
        Exclude halted stocks on rebalance dates.
    reb_freq : str
        Rebalance frequency used for universe selection.

    Returns
    -------
    pd.DataFrame
        Boolean mask where True means the asset is included in the universe.
    """
    close = _prepare_daily_df(close)
    cap = _prepare_daily_df(cap)
    volume = _prepare_daily_df(volume)
    shares_outstanding = _prepare_daily_df(shares_outstanding)
    halted = _prepare_daily_df(halted)

    common_cols = sorted(
        set(close.columns)
        & set(cap.columns)
        & set(volume.columns)
        & set(shares_outstanding.columns)
        & set(halted.columns)
    )
    close = close[common_cols]
    cap = cap[common_cols]
    volume = volume[common_cols]
    shares_outstanding = shares_outstanding[common_cols]
    halted = halted[common_cols]

    mask = pd.DataFrame(True, index=close.index, columns=common_cols)
    reb_dates = pd.to_datetime(_rebalance_dates(close.index, reb_freq))

    turnover = volume.div(shares_outstanding).replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    monthly_turnover = turnover.resample("M").mean()

    for dt in reb_dates:
        if dt not in mask.index:
            continue

        prev_month_end = _previous_month_end(dt)

        # Market cap filter: use the last available cap value through prior month end.
        try:
            cap_prev = cap.loc[:prev_month_end].iloc[-1]
        except IndexError:
            cap_prev = pd.Series(index=common_cols, dtype=float)

        cap_excluded = cap_prev.isna() | (cap_prev < cap_threshold)
        mask.loc[dt, cap_excluded.index] = mask.loc[dt, cap_excluded.index] & ~cap_excluded.fillna(True)

        # Turnover filter: exclude worst-performing names from prior month.
        try:
            turnover_prev = monthly_turnover.loc[prev_month_end]
        except KeyError:
            turnover_prev = pd.Series(index=common_cols, dtype=float)

        valid_turnover = turnover_prev.dropna()
        if not valid_turnover.empty and 0 < turnover_quantile < 1:
            cutoff = valid_turnover.quantile(turnover_quantile)
            turnover_excluded = turnover_prev <= cutoff
            turnover_excluded = turnover_excluded.reindex(common_cols, fill_value=False)
            mask.loc[dt, turnover_excluded.index] = mask.loc[dt, turnover_excluded.index] & ~turnover_excluded

        # Halted filter: exclude assets halted on the rebalance date.
        if exclude_halted:
            halted_row = halted.reindex(mask.index).loc[dt]
            halted_excluded = halted_row.fillna(0).astype(float) >= 0.5
            mask.loc[dt, halted_excluded.index] = mask.loc[dt, halted_excluded.index] & ~halted_excluded

    return mask


def apply_universe_mask(signal: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Apply a universe mask to a factor signal frame.

    Non-included assets are set to NaN.
    """
    signal_df = _prepare_daily_df(signal)
    signal_df = signal_df.reindex(index=mask.index, columns=mask.columns)
    result = signal_df.where(mask)
    result = result.reset_index()
    return result


def filter_signal(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    cap: pd.DataFrame,
    volume: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
    halted: pd.DataFrame,
    cap_threshold: float = 100000.0,
    turnover_quantile: float = 0.05,
    exclude_halted: bool = True,
    reb_freq: str = "M"
) -> pd.DataFrame:
    mask = build_universe_mask(
        close=close,
        cap=cap,
        volume=volume,
        shares_outstanding=shares_outstanding,
        halted=halted,
        cap_threshold=cap_threshold,
        turnover_quantile=turnover_quantile,
        exclude_halted=exclude_halted,
        reb_freq=reb_freq,
    )
    return apply_universe_mask(signal, mask)
