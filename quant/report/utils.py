from typing import Dict, Tuple

import pandas as pd

from quant.analytics import sharpe


def compute_monthly_returns(returns: pd.Series) -> pd.Series:
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    return returns.resample("M").apply(lambda x: (1 + x).prod() - 1).dropna()


def compute_phase_labels(benchmark_cum: pd.Series) -> pd.Series:
    if benchmark_cum is None or benchmark_cum.empty:
        return pd.Series(dtype=object)
    monthly_benchmark = benchmark_cum.resample("M").last().dropna()
    rolling_12m = monthly_benchmark.pct_change(12)

    def label(value: float) -> str:
        if value > 0.10:
            return "Bull"
        if value < -0.10:
            return "Bear"
        return "Sideways"

    return rolling_12m.map(label).dropna()


def compute_phase_summary(returns: pd.Series, phase_labels: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if returns is None or returns.empty or phase_labels is None or phase_labels.empty:
        return pd.DataFrame(), pd.DataFrame()

    monthly_returns = returns.to_frame(name="return").dropna()
    aligned_phase_labels = phase_labels.reindex(monthly_returns.index, method="ffill").dropna()
    monthly_returns = monthly_returns.loc[aligned_phase_labels.index]

    rows = []
    for phase in aligned_phase_labels.unique():
        mask = aligned_phase_labels == phase
        phase_ret = monthly_returns.loc[mask, "return"]
        if phase_ret.empty:
            continue
        annualized = (1 + phase_ret).prod() ** (12 / len(phase_ret)) - 1
        rows.append({
            "Phase": phase,
            "Months": len(phase_ret),
            "CAGR": annualized,
            "Sharpe": sharpe(phase_ret, freq="M"),
        })

    phase_summary = pd.DataFrame(rows).set_index("Phase")
    monthly_phase = pd.DataFrame({"Monthly Return": monthly_returns["return"], "Phase": aligned_phase_labels})
    return phase_summary, monthly_phase


def build_comparison_performance_table(strategy_summary: Dict[str, float], benchmark_summary: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame({
        "Strategy": pd.Series(strategy_summary),
        "Benchmark": pd.Series(benchmark_summary),
    })
    df["Excess"] = df["Strategy"] - df["Benchmark"]
    return df


def build_phase_comparison_table(strategy_phase: pd.DataFrame, benchmark_phase: pd.DataFrame) -> pd.DataFrame:
    idx = strategy_phase.index.union(benchmark_phase.index)
    strategy = strategy_phase.reindex(idx)
    benchmark = benchmark_phase.reindex(idx)

    rows = []
    for phase in idx:
        rows.append({
            "Phase": phase,
            "Strategy Months": strategy.loc[phase, "Months"],
            "Benchmark Months": benchmark.loc[phase, "Months"],
            "Strategy CAGR": strategy.loc[phase, "CAGR"],
            "Benchmark CAGR": benchmark.loc[phase, "CAGR"],
            "Excess CAGR": strategy.loc[phase, "CAGR"] - benchmark.loc[phase, "CAGR"],
            "Strategy Sharpe": strategy.loc[phase, "Sharpe"],
            "Benchmark Sharpe": benchmark.loc[phase, "Sharpe"],
            "Excess Sharpe": strategy.loc[phase, "Sharpe"] - benchmark.loc[phase, "Sharpe"],
        })

    return pd.DataFrame(rows).set_index("Phase")


def build_annual_return_table(portfolio_ret: pd.Series, benchmark_ret: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Portfolio": portfolio_ret, "Benchmark": benchmark_ret}).dropna()
    df = df.groupby(df.index.to_period("Y")).apply(lambda x: (1 + x).prod() - 1)
    df.index = df.index.year
    df["Excess"] = df["Portfolio"] - df["Benchmark"]
    return df


def build_report_overview(config: Dict, benchmark_cum: pd.Series, summary: Dict[str, float], benchmark_summary: Dict[str, float]) -> str:
    if config["group"] == "Sector":
        selection_text = (
            f"시장 섹터 비중을 추종하며 섹터 내 {config['direction']} {config['factor']} 종목을 "
            f"상위 {config['top_pct']:.0%}로 1차 후보군에 포함합니다."
        )
    else:
        selection_text = (
            f"섹터 구분 없이 전체 시장에서 {config['direction']} {config['factor']} 종목을 "
            f"상위 {config['top_pct']:.0%}로 1차 후보군에 포함합니다."
        )

    if config["allocation"] == "signal":
        allocation_text = (
            f"1차 후보군에서 선정된 종목은 {config['weighting']} 신호 기반 비중으로 편입하며, "
            f"최대 종목 비중은 {config['max_weight']:.2%}로 제한합니다."
        )
    else:
        allocation_text = (
            f"1차 후보군에서 선정된 종목은 {config['allocation']} 방식으로 비중을 최적화하며, "
            f"최대 종목 비중은 {config['max_weight']:.2%}로 제한합니다."
        )

    cagr = summary.get("CAGR", float("nan"))
    sharpe_val = summary.get("Sharpe", float("nan"))
    mdd = summary.get("MDD", float("nan"))

    benchmark_cagr = float("nan")
    if benchmark_cum is not None and not benchmark_cum.empty and len(benchmark_cum) > 1:
        years = (benchmark_cum.index[-1] - benchmark_cum.index[0]).days / 365.25
        benchmark_cagr = benchmark_cum.iloc[-1] ** (1 / years) - 1 if years > 0 else float("nan")

    relative = "우수한" if cagr > benchmark_cagr else "낮은"
    performance_text = (
        f"백테스트 기간 해당 전략의 CAGR은 {cagr:.2%}로 코스피 대비 {relative} 성과를 기록했으며 "
        f"MDD는 {mdd:.2%}로 집계되었습니다. 샤프지수는 {sharpe_val:.2f}로 평가됩니다."
    )

    return (
        "## 요약\n\n"
        "이 리포트는 KOSPI 기반 유니버스를 사용하여 하나의 전략을 백테스트하고 전략 개요, 성과 요약, 국면별 분석 및 초과 수익률을 제공합니다.\n\n"
        f"{selection_text} {allocation_text}\n\n"
        f"{performance_text}"
    )
