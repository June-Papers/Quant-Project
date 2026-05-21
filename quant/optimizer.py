"""Portfolio allocation methods for optimized equity weighting."""

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .utils import apply_max_weight_limit


def _regularize_cov(cov: pd.DataFrame, min_eig: float = 1e-6) -> pd.DataFrame:
    cov = cov.copy().astype(float)
    if cov.shape[0] == 0:
        return cov

    try:
        vals, vecs = np.linalg.eigh(cov.values)
    except np.linalg.LinAlgError:
        return cov

    vals = np.clip(vals, min_eig, None)
    return pd.DataFrame(vecs @ np.diag(vals) @ vecs.T, index=cov.index, columns=cov.columns)


def equal_weight_allocation(assets: Iterable[str], max_weight: Optional[float] = None) -> pd.Series:
    assets = list(assets)
    if len(assets) == 0:
        return pd.Series(dtype=float)

    weight = pd.Series(1.0 / len(assets), index=assets)
    if max_weight is not None:
        weight = apply_max_weight_limit(weight, max_weight)
    return weight


def markowitz_allocation(
    returns: pd.DataFrame,
    signals: pd.Series,
    direction: str = "low",
    max_weight: Optional[float] = None
) -> pd.Series:
    if returns.empty or signals.empty:
        return pd.Series(dtype=float)

    cov = returns.cov()
    cov = _regularize_cov(cov)
    if cov.isnull().values.any():
        cov = cov.fillna(0.0)

    mu = signals.reindex(cov.columns).fillna(0.0).astype(float)
    if direction == "low":
        mu = -mu

    try:
        w_raw = np.linalg.solve(cov.values, mu.values)
    except np.linalg.LinAlgError:
        w_raw, *_ = np.linalg.lstsq(cov.values, mu.values, rcond=None)

    w = np.maximum(w_raw, 0.0)
    if w.sum() == 0:
        w = np.ones_like(w)

    weights = pd.Series(w / w.sum(), index=cov.columns)
    if max_weight is not None:
        weights = apply_max_weight_limit(weights, max_weight)
    return weights


def risk_parity_allocation(
    returns: pd.DataFrame,
    max_weight: Optional[float] = None,
    max_iter: int = 200,
    tol: float = 1e-6
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)

    cov = returns.cov()
    cov = _regularize_cov(cov)
    if cov.isnull().values.any():
        cov = cov.fillna(0.0)

    n = cov.shape[0]
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov.columns)

    w = np.ones(n) / n
    for _ in range(max_iter):
        last = w.copy()
        sigma = cov.values.dot(w)
        denom = sigma + 1e-12
        w = w / denom
        if w.sum() == 0:
            w = np.ones(n) / n
        else:
            w = w / w.sum()
        if np.linalg.norm(w - last) < tol:
            break

    weights = pd.Series(w, index=cov.columns)
    if max_weight is not None:
        weights = apply_max_weight_limit(weights, max_weight)
    return weights
