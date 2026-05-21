"""Utility helpers for the quant package."""
from typing import List
import pandas as pd
import numpy as np


def align_dataframes(dfs: List[pd.DataFrame]):
    """Align list of dataframes that have a 'date' column and stock columns.

    Returns list of DataFrames indexed by datetime with common cols and dates.
    """
    aligned = []

    for df in dfs:
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]) if "date" in tmp.columns else tmp.index
        tmp = tmp.sort_values("date").set_index("date")
        aligned.append(tmp)

    # common cols
    common_cols = set(aligned[0].columns)
    for df in aligned[1:]:
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)

    aligned = [df[common_cols] for df in aligned]

    # common dates
    common_dates = aligned[0].index
    for df in aligned[1:]:
        common_dates = common_dates.intersection(df.index)

    aligned = [df.loc[common_dates] for df in aligned]

    return aligned


def get_rebalance_dates(index: pd.DatetimeIndex, reb_freq: str = "M") -> List[pd.Timestamp]:
    reb_freq = reb_freq.upper()
    if reb_freq == "D":
        return index.values
    if reb_freq in ("W", "M", "Q", "Y"):
        return (
            index.to_series()
            .groupby(index.to_period(reb_freq))
            .first()
            .values
        )
    raise ValueError("reb_freq must be 'D','W','M','Q' or 'Y'")


def apply_max_weight_limit(weight: pd.Series, max_weight: float, max_iter: int = 100, tol: float = 1e-8) -> pd.Series:
    """Apply max weight limit with efficient iterative algorithm.
    
    Uses bisection method to find the optimal multiplier that satisfies all constraints.
    Much faster than sequential redistribution approach.
    """
    w = weight.copy().astype(float)
    total = w.sum()
    
    if total == 0 or len(w) == 0:
        return w
    
    # Normalize to sum to 1
    w = w / total
    
    # If all weights are already below max_weight, return
    if w.max() <= max_weight * (1 + tol):
        return w
    
    # Use binary search to find optimal multiplier λ
    # We need to find λ such that: min(w[i] * λ, max_weight) sums to 1
    lambda_min = 0.0
    lambda_max = 1.0
    
    for _ in range(max_iter):
        lambda_mid = (lambda_min + lambda_max) / 2.0
        
        # Apply multiplier with cap
        w_test = np.minimum(w * lambda_mid, max_weight)
        w_sum = w_test.sum()
        
        if w_sum == 0:
            lambda_max = lambda_mid
            continue
            
        if abs(w_sum - 1.0) < tol:
            return pd.Series(w_test, index=weight.index)
        elif w_sum < 1.0:
            lambda_min = lambda_mid
        else:
            lambda_max = lambda_mid
    
    # Final application with optimal lambda.
    # If the cap makes a full investment impossible, return capped weights
    # without renormalizing, so max_weight stays respected.
    w_result = np.minimum(w * ((lambda_min + lambda_max) / 2.0), max_weight)
    return pd.Series(w_result, index=weight.index)
