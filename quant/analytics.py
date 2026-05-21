"""Performance analytics and simple plotting helpers."""
import numpy as np
import pandas as pd


def CAGR(cum_return: pd.Series) -> float:
    if len(cum_return) < 2:
        return np.nan
    start, end = cum_return.index[0], cum_return.index[-1]
    years = (end - start).days / 365.25
    return cum_return.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan


def sharpe(portfolio_return: pd.Series, freq: str = "D", riskfree: float = 0.0) -> float:
    # annualize factor
    freq_map = {"D": 252, "W": 52, "M": 12}
    ann = freq_map.get(freq.upper(), 252)
    r = portfolio_return.dropna()
    if r.std() == 0:
        return np.nan
    return (r.mean() - riskfree / ann) / r.std() * np.sqrt(ann)


def max_drawdown(cum_return: pd.Series) -> float:
    peak = cum_return.cummax()
    draw = (cum_return / peak) - 1
    return draw.min()


def performance_summary(result: dict) -> dict:
    port_ret = result["portfolio_return"].fillna(0)
    cum = result["cum_return"]
    summary = {
        "CAGR": CAGR(cum),
        "Sharpe": sharpe(port_ret),
        "MDD": max_drawdown(cum),
        "Turnover(mean)": result["turnover"].mean(),
    }
    return summary
