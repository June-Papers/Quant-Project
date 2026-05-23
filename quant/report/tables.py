from typing import Dict, List
import pandas as pd


def build_strategy_overview(configs: List[Dict]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        rows.append({
            "Strategy": cfg["name"],
            "Universe": cfg["universe_description"],
            "Factor": cfg["factor"],
            "Group": cfg["group"],
            "Direction": cfg["direction"],
            "Weighting": cfg["weighting"],
            "Allocation": cfg["allocation"],
            "Max Weight": cfg["max_weight"],
            "Rebalance": cfg["reb_freq"],
            "Transaction Cost": cfg["transaction_cost"],
        })
    return pd.DataFrame(rows)


def build_performance_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(metrics).T.loc[:, ["CAGR", "Sharpe", "MDD"]]


def build_phase_performance_table(phase_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for phase, stats in phase_metrics.items():
        row = {"Phase": phase}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Phase")


def build_factor_excess_table(portfolio_ret: pd.Series, benchmark_ret: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        "Portfolio": portfolio_ret,
        "Benchmark": benchmark_ret,
    }).dropna()
    df["Excess"] = df["Portfolio"] - df["Benchmark"]
    return df
