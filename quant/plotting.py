from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _valid_date_range(series_map):
    start_dates = [series.dropna().index.min() for series in series_map.values() if not series.dropna().empty]
    end_dates = [series.dropna().index.max() for series in series_map.values() if not series.dropna().empty]
    if not start_dates or not end_dates:
        return None, None
    return max(start_dates), min(end_dates)


def save_equity_curve(series_map, path: Path, title: str = "Equity Curve") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    start, end = _valid_date_range(series_map)
    for label, series in series_map.items():
        series = series.dropna()
        if start is not None and end is not None:
            series = series[(series.index >= start) & (series.index <= end)]
        ax.plot(series.index, series.values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_drawdown_curve(series, path: Path, title: str = "Drawdown") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    series = series.dropna()
    peak = series.cummax()
    drawdown = series / peak - 1
    ax.plot(series.index, drawdown.values, color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_drawdown_curves(strategy_series, benchmark_series, path: Path, title: str = "Drawdown") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    series_map = {"Strategy": strategy_series, "Benchmark": benchmark_series}
    start, end = _valid_date_range(series_map)
    for label, series in series_map.items():
        series = series.dropna()
        if start is not None and end is not None:
            series = series[(series.index >= start) & (series.index <= end)]
        peak = series.cummax()
        drawdown = series / peak - 1
        ax.plot(series.index, drawdown.values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_excess_return_bar_chart(df: pd.DataFrame, path: Path, title: str = "Factor Excess Return") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    df = df.dropna()
    ax.bar(df.index.astype(str), df["Excess"].values, color="tab:blue", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Excess Return")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_phase_performance_bar(df: pd.DataFrame, path: Path, title: str = "Phase Performance") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = range(len(df))
    for idx, col in enumerate(df.columns):
        ax.bar([p + width * idx for p in x], df[col].values, width=width, label=col)
    ax.set_xticks([p + width * (len(df.columns) - 1) / 2 for p in x])
    ax.set_xticklabels(df.index)
    ax.set_title(title)
    ax.set_ylabel("Annualized Return / Sharpe")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_excess_return_chart(df: pd.DataFrame, path: Path, title: str = "Factor Excess Return") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df.columns:
        ax.plot(df.index, df[col].values, label=col)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
