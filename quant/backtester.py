"""Portfolio construction and backtesting engine.""" 
from typing import Optional
import pandas as pd
import numpy as np
from .utils import ( align_dataframes, get_rebalance_dates, apply_max_weight_limit)
from .optimizer import (
    equal_weight_allocation,
    markowitz_allocation,
    risk_parity_allocation,
)

class Backtester:
    def __init__(self, close, cap=None, sector=None):
        self.close = close.copy()
        self.cap = cap.copy() if cap is not None else None
        self.sector = sector.copy() if sector is not None else None
        
        # Cache for historical returns
        self._returns_cache = {}
        self._close_index = None
        self._precompute_returns()

    # =====================================================
    # CACHE INITIALIZATION
    # =====================================================
    def _precompute_returns(self):
        """Precompute and cache returns to avoid repeated calculations."""
        close = self.close.copy()
        try:
            close = close.set_index("date")
        except Exception:
            pass
        
        close = close.sort_index()
        self._returns_cache = close.pct_change().dropna(how="all")
        self._close_index = close.index

    # =====================================================
    # SCORE (same logic, safer)
    # =====================================================
    def make_score(self, signal, direction="low", weighting="score", top_pct=0.2):

        signal = signal.dropna()
        if len(signal) == 0:
            return pd.Series(dtype=float)

        raw = -signal if direction == "low" else signal

        if weighting == "score":
            std = raw.std()
            std = std if std != 0 else 1
            score = (raw - raw.mean()) / std
            return score.clip(lower=0)

        elif weighting == "rank":
            # Compute rank-based score where higher is better regardless of direction
            n = len(signal)
            # rank: 1..n (1 = smallest signal)
            rank = signal.rank(method="average")

            if direction == "low":
                # smaller signal is better -> invert rank so best ~= n
                rank_score = (n - rank + 1) / n
            else:
                # larger signal is better
                rank_score = rank / n

            # apply top_pct cutoff (keep only top fraction)
            cutoff = 1 - top_pct
            score = rank_score.where(rank_score >= cutoff, 0)
            return score / score.max() if score.max() != 0 else score

        elif weighting == "equal":
            return pd.Series(1.0, index=signal.index)

        else:
            raise ValueError("invalid weighting")

    def select_stocks_for_portfolio(self, signal_row, cap_row, sector_row,
                                    group, direction, weighting, top_n, top_pct):
        signal_row = signal_row.dropna()
        if signal_row.empty:
            return pd.Series(dtype=float)

        if group == "Market":
            selected = self.select_stocks(signal_row, direction, top_n)
            score = self.make_score(selected, direction, weighting, top_pct)
            return selected.loc[score[score > 0].index]

        final = pd.Series(dtype=float)
        for sec in sector_row.dropna().unique():
            stocks = sector_row[sector_row == sec].index
            stocks = stocks.intersection(signal_row.index)
            if len(stocks) == 0:
                continue

            sec_signal = signal_row.loc[stocks].dropna()
            if sec_signal.empty:
                continue

            sec_signal = self.select_stocks(sec_signal, direction, top_n)
            score = self.make_score(sec_signal, direction, weighting, top_pct)
            keep = score[score > 0].index
            if len(keep) == 0:
                continue

            final = pd.concat([final, sec_signal.loc[keep]])

        return final

    def get_historical_returns(self, dt, assets, lookback=252):
        """Get historical returns using pre-computed cache."""
        # Get returns up to dt from cache
        returns = self._returns_cache.loc[:dt]
        returns = returns.reindex(columns=assets).dropna(axis=1, how="all")
        return returns.tail(lookback)

    def compute_optimized_weights(self, selected_signal, dt, portfolio_method, max_weight, direction):
        assets = selected_signal.index.tolist()
        if len(assets) == 0:
            return pd.Series(dtype=float)

        returns = self.get_historical_returns(dt, assets)
        if portfolio_method == "equal_weight":
            return equal_weight_allocation(assets, max_weight)

        if returns.shape[1] == 0:
            return equal_weight_allocation(assets, max_weight)

        if portfolio_method == "risk_parity":
            return risk_parity_allocation(returns, max_weight)

        if portfolio_method == "markowitz":
            return markowitz_allocation(returns, selected_signal, direction, max_weight)

        raise ValueError(f"Unknown portfolio_method: {portfolio_method}")

    # =====================================================
    # STOCK SELECTION (safe version)
    # =====================================================
    def select_stocks(self, signal, direction="low", top_n=None):
        signal = signal.dropna()
        if top_n is None:
            return signal

        if direction == "low":
            idx = signal.nsmallest(top_n).index
        else:
            idx = signal.nlargest(top_n).index

        return signal.loc[idx]

    # =====================================================
    # MARKET WEIGHT
    # =====================================================
    def make_market_weight(self, signal_row, direction, weighting, top_n, top_pct, max_weight):

        signal_row = signal_row.dropna()
        signal_row = self.select_stocks(signal_row, direction, top_n)

        score = self.make_score(signal_row, direction, weighting, top_pct)

        if score.sum() == 0:
            return pd.Series(0, index=signal_row.index)

        weight = score / score.sum()
        return apply_max_weight_limit(weight, max_weight)

    # =====================================================
    # SECTOR WEIGHT (safe indexing fix)
    # =====================================================
    def make_sector_weight(self, signal_row, cap_row, sector_row,
                           direction, weighting, top_n, top_pct, max_weight):

        final = pd.Series(0.0, index=signal_row.index)

        total_cap = cap_row.sum() if cap_row.sum() > 0 else 1

        for sec in sector_row.dropna().unique():

            stocks = sector_row[sector_row == sec].index

            sec_signal = signal_row.loc[stocks].dropna()
            sec_cap = cap_row.loc[stocks]

            if len(sec_signal) == 0:
                continue

            sec_signal = self.select_stocks(sec_signal, direction, top_n)
            sec_cap = sec_cap.loc[sec_signal.index]

            score = self.make_score(sec_signal, direction, weighting, top_pct)

            if score.sum() == 0:
                continue

            w = score / score.sum()
            w *= sec_cap.sum() / total_cap

            final.loc[w.index] = w

        return apply_max_weight_limit(final, max_weight)

    # =====================================================
    # WEIGHT MATRIX
    # =====================================================
    def make_weight(self, signal, group="Market", direction="low",
                    weighting="score", top_n=None, top_pct=0.2,
                    max_weight=0.05, reb_freq="M", portfolio_method="signal"):

        if group == "Sector":
            if self.cap is None or self.sector is None:
                raise ValueError("Sector mode requires cap + sector")

            signal, cap, sector = align_dataframes([signal, self.cap, self.sector])
        else:
            signal, _ = align_dataframes([signal, self.close])
            cap = sector = None

        weights = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)

        reb_dates = get_rebalance_dates(signal.index, reb_freq)

        for dt in reb_dates:

            row = signal.loc[dt]

            # skip if no valid signals on this rebalance date
            if row.dropna().empty:
                continue

            if portfolio_method == "signal":
                if group == "Market":
                    w = self.make_market_weight(row, direction, weighting, top_n, top_pct, max_weight)
                else:
                    w = self.make_sector_weight(row, cap.loc[dt], sector.loc[dt],
                                                direction, weighting, top_n, top_pct, max_weight)
            else:
                selected = self.select_stocks_for_portfolio(
                    row, cap.loc[dt] if cap is not None else None,
                    sector.loc[dt] if sector is not None else None,
                    group, direction, weighting, top_n, top_pct
                )
                if selected.empty:
                    continue
                w = self.compute_optimized_weights(selected, dt, portfolio_method, max_weight, direction)

            # Ensure the row starts from zero so previous weights don't carry through
            weights.loc[dt, :] = 0

            # Safety: ensure weights are numeric, normalized and respect max limit
            try:
                w = w.astype(float)
            except Exception:
                pass
            if isinstance(w, pd.Series) and w.sum() != 0:
                w = w / w.sum()
            if isinstance(w, pd.Series):
                w = apply_max_weight_limit(w, max_weight)

            weights.loc[dt, w.index] = w

        # Ensure weights cover the full trading calendar of `self.close`.
        try:
            close_idx = self.close.set_index('date').index
            weights = weights.reindex(close_idx)
        except Exception:
            # fallback: keep existing index if reindexing fails
            pass

        return weights.ffill().fillna(0)

    # =====================================================
    # BACKTEST ENGINE
    # =====================================================
    def run_backtest(self, weights, transaction_cost=0.0025):

        # Ensure weights index is a datetime index named 'date' so align_dataframes
        # can detect and align by the 'date' column correctly.
        weights = weights.copy()
        try:
            weights.index = pd.to_datetime(weights.index)
        except Exception:
            pass
        weights.index.name = 'date'

        weights, close = align_dataframes([weights.reset_index(), self.close])

        ret = close.pct_change()

        port_ret = (weights.shift(1) * ret).sum(axis=1)

        turnover = weights.diff().abs().sum(axis=1)
        turnover.iloc[0] = weights.iloc[0].abs().sum()

        port_ret = port_ret - turnover * transaction_cost

        cum = (1 + port_ret.fillna(0)).cumprod()

        return {
            "portfolio_return": port_ret,
            "cum_return": cum,
            "weights": weights,
            "turnover": turnover
        }

    # =====================================================
    # MAIN ENTRY
    # =====================================================
    def backtest(self, signal, transaction_cost=0.0025,
                 group="Market", direction="low",
                 weighting="score", top_n=None,
                 top_pct=0.2, max_weight=0.05,
                 reb_freq="M", portfolio_method="signal"):

        weights = self.make_weight(
            signal, group, direction,
            weighting, top_n, top_pct,
            max_weight, reb_freq, portfolio_method
        )

        return self.run_backtest(weights, transaction_cost)