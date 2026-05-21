"""Factor construction and preprocessing utilities."""
from typing import Optional
import pandas as pd
import numpy as np
from .utils import align_dataframes


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.isna().all():
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore_series(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) or 1)


def neutralize_series(factor: pd.Series, exposure: pd.Series) -> pd.Series:
    """Simple linear neutralization of factor against a single exposure (cross-section).

    Both inputs are Series indexed by asset.
    Returns residual series (mean centered).
    """
    valid = factor.notna() & exposure.notna()
    if valid.sum() == 0:
        return factor
    x = exposure[valid].values.reshape(-1, 1)
    y = factor[valid].values
    # add intercept
    X = np.hstack([np.ones((x.shape[0], 1)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X.dot(coef)
    resid = y - fitted
    out = factor.copy()
    out.loc[valid] = resid
    return out - out.mean()


# ============================================
# Factor Builders
# ============================================
def build_pbr(dl, close, bs):

    book = dl.get_financial_data(
        df=bs,
        account_name="자산총계(천원)",
        close_df=close,
        apply_lag=True,
        daily_fill=True
    )

    price = close.set_index("date")
    book = book.set_index("date")

    common_cols = list(
        set(price.columns) &
        set(book.columns)
    )

    factor = (
        price[common_cols] /
        book[common_cols].replace(0, np.nan)
    )

    return factor.reset_index()



def build_per(dl, close, pl):

    ni = dl.get_financial_data(
        df=pl,
        account_name="당기순이익(천원)",
        close_df=close,
        apply_lag=True,
        daily_fill=True
    )

    price = close.set_index("date")
    ni = ni.set_index("date")

    common_cols = list(
        set(price.columns) &
        set(ni.columns)
    )

    factor = (
        price[common_cols] /
        ni[common_cols].replace(0, np.nan)
    )

    return factor.reset_index()


def build_roe(dl, close, bs, pl):
    equity = dl.get_financial_data(
        df=bs,
        account_name="자본총계(천원)",
        close_df=close,
        apply_lag=True,
        daily_fill=True
    )

    ni = dl.get_financial_data(
        df=pl,
        account_name="당기순이익(천원)",
        close_df=close,
        apply_lag=True,
        daily_fill=True
    )

    equity = equity.set_index("date")
    ni = ni.set_index("date")

    common_cols = list(
        set(equity.columns) &
        set(ni.columns)
    )

    factor = (
        ni[common_cols] /
        equity[common_cols].replace(0, np.nan)
    )

    return factor.reset_index()