"""Quant research package - lightweight entrypoints."""

from .backtester import Backtester
from .data import DataLoader
from .factor import (
    winsorize_series,
    zscore_series,
    neutralize_series,
    build_pbr,
    build_per,
    build_roe,
)
from .analytics import (
    performance_summary,
    CAGR,
    sharpe,
    max_drawdown,
)
from .universe import (
    build_universe_mask,
    apply_universe_mask,
    filter_signal,
)

__all__ = [
    "Backtester",
    "DataLoader",
    "winsorize_series",
    "zscore_series",
    "neutralize_series",
    "build_pbr",
    "build_per",
    "build_roe",
    "build_universe_mask",
    "apply_universe_mask",
    "filter_signal",
    "performance_summary",
    "CAGR",
    "sharpe",
    "max_drawdown",
]
