from pathlib import Path
from typing import Dict
import pandas as pd
from ..plotting import (
    save_equity_curve,
    save_drawdown_curves,
    save_phase_performance_bar,
    save_excess_return_bar_chart,
)


def generate_report_charts(
    output_dir: Path,
    strategy_cum_returns: Dict[str, pd.Series],
    benchmark: Dict[str, pd.Series],
    phase_cagr: pd.DataFrame,
    factor_excess: Dict[str, pd.DataFrame],
    strategy_cum_return: pd.Series,
    benchmark_cum_return: pd.Series,
    annual_excess_returns: pd.DataFrame,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_path = output_dir / "equity_curve.png"
    save_equity_curve(
        {
            **strategy_cum_returns,
            "Benchmark": benchmark["cum_return"],
        },
        equity_path,
        title="Strategies vs Benchmark Equity Curve",
    )

    drawdown_path = output_dir / "drawdown.png"
    save_drawdown_curves(
        strategy_cum_return,
        benchmark_cum_return,
        drawdown_path,
        title="Strategy and Benchmark Drawdown",
    )

    phase_path = output_dir / "phase_performance.png"
    if not phase_cagr.empty:
        save_phase_performance_bar(phase_cagr, phase_path, title="Phase CAGR by Strategy vs Benchmark")

    factor_paths = {}
    for factor_name, df in factor_excess.items():
        chart_path = output_dir / f"{factor_name.lower().replace(' ', '_')}_excess.png"
        save_excess_return_bar_chart(df, chart_path, title=f"{factor_name} Yearly Excess Returns")
        factor_paths[factor_name] = chart_path

    return {
        "equity_curve": equity_path,
        "drawdown": drawdown_path,
        "phase_performance": phase_path,
        **factor_paths,
    }
