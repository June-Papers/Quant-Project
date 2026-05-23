import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from quant.data import DataLoader
from quant.backtester import Backtester
from quant.analytics import performance_summary, sharpe
from quant.universe import filter_signal
import quant.factor as factor_module
from quant.benchmark import build_market_benchmark
from quant.report import (
    ReportAssembler,
    generate_report_charts,
    build_strategy_overview,
    build_performance_table,
    build_phase_performance_table,
    build_factor_excess_table,
)


def build_strategy_configs(args) -> List[Dict]:
    universe_description = (
        f"KOSPI universe filtered by market cap >= {args.cap_threshold:.0f}백만원 "
        f"and excluding lowest {args.turnover_quantile * 100:.1f}% liquidity names."
    )

    return [
        {
            "name": "Strategy 1",
            "factor": "pbr",
            "group": "Sector",
            "direction": "low",
            "weighting": "rank",
            "allocation": "signal",
            "top_pct": 0.2,
            "top_n": None,
            "max_weight": 0.05,
            "reb_freq": args.reb_freq,
            "transaction_cost": args.transaction_cost,
            "universe_description": universe_description,
            "description": (
                "Sector-weighted value strategy: 시장 섹터 비중을 추종하며 PBR 하위 20%를 rank 방식으로 선별합니다."
            ),
        },
        {
            "name": "Strategy 2",
            "factor": "roe",
            "group": "Sector",
            "direction": "high",
            "weighting": "score",
            "allocation": "markowitz",
            "top_pct": 0.15,
            "top_n": None,
            "max_weight": 0.08,
            "reb_freq": args.reb_freq,
            "transaction_cost": args.transaction_cost,
            "universe_description": universe_description,
            "description": (
                "Sector 비중 추종 전략: 고ROE 상위 15%를 score 방식으로 선별하고 Markowitz 최적화로 배분합니다."
            ),
        },
    ]


def build_factor(dl, config, close, bs, pl):
    if config["factor"] == "pbr":
        return factor_module.build_pbr(dl, close, bs)
    if config["factor"] == "per":
        return factor_module.build_per(dl, close, pl)
    if config["factor"] == "roe":
        return factor_module.build_roe(dl, close, bs, pl)
    raise ValueError(f"Unsupported factor: {config['factor']}")


def compute_phase_labels(benchmark_cum):
    monthly = benchmark_cum.resample("M").last().dropna()
    rolling_12m = monthly.pct_change(12)
    def phase_label(val):
        if val > 0.10:
            return "Bull"
        if val < -0.10:
            return "Bear"
        return "Sideways"
    return rolling_12m.map(phase_label).dropna()


def compute_phase_performance(
    returns: pd.Series,
    phase_labels: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if returns.empty or phase_labels.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = pd.DataFrame({"return": returns}).dropna()
    phases = phase_labels.reindex(dates.index, method="ffill").dropna()
    table_rows = []
    for phase in phases.unique():
        mask = phases == phase
        phase_ret = dates.loc[mask, "return"]
        if phase_ret.empty:
            continue
        annualized = (1 + phase_ret).prod() ** (252 / len(phase_ret)) - 1
        table_rows.append({"Phase": phase, "Months": int(mask.sum() / 21), "CAGR": annualized, "Sharpe": sharpe(phase_ret, freq="D")})

    phase_df = pd.DataFrame(table_rows).set_index("Phase")
    return phase_df, phase_df


def align_returns(portfolio_ret: pd.Series, benchmark_ret: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Portfolio": portfolio_ret, "Benchmark": benchmark_ret}).dropna()
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate quant strategy report")
    parser.add_argument("--data-path", type=str, default="../data", help="Path to source parquet data")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for report files")
    parser.add_argument("--cap-threshold", type=float, default=100000.0, help="Universe market cap threshold in 백만원")
    parser.add_argument("--turnover-quantile", type=float, default=0.05, help="Exclude lowest liquidity quantile")
    parser.add_argument("--reb-freq", type=str, default="M", choices=["D", "W", "M", "Q", "Y"], help="Rebalance frequency")
    parser.add_argument("--transaction-cost", type=float, default=0.0025, help="Transaction cost per turnover")
    parser.add_argument("--report-name", type=str, default="strategy_report", help="Base report filename")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading market and financial data...")
    dl = DataLoader(data_path=args.data_path)
    close = dl.load_close()
    cap = dl.load_cap()
    sector = dl.load_sector()
    volume = dl.load_volume()
    shares_outstanding = dl.load_shares_outstanding()
    halted = dl.load_halted()
    bs = dl.load_bs()
    pl = dl.load_pl()

    print("Building benchmark series...")
    benchmark = build_market_benchmark(close, cap)

    configs = build_strategy_configs(args)
    backtester = Backtester(close=close, cap=cap, sector=sector)

    portfolio_results = {}
    factor_excess = {}
    phase_cagr_dict = {}
    summary_metrics = {}

    phase_labels = compute_phase_labels(benchmark["cum_return"])

    for config in configs:
        name = config["name"]
        print(f"Processing {name}...")
        factor = build_factor(dl, config, close, bs, pl)
        signal = filter_signal(
            signal=factor,
            close=close,
            cap=cap,
            volume=volume,
            shares_outstanding=shares_outstanding,
            halted=halted,
            cap_threshold=args.cap_threshold,
            turnover_quantile=args.turnover_quantile,
            exclude_halted=True,
            reb_freq=args.reb_freq,
        )

        result = backtester.backtest(
            signal=signal,
            transaction_cost=config["transaction_cost"],
            group=config["group"],
            direction=config["direction"],
            weighting=config["weighting"],
            top_n=config["top_n"],
            top_pct=config["top_pct"],
            max_weight=config["max_weight"],
            reb_freq=config["reb_freq"],
            portfolio_method=config["allocation"],
        )

        portfolio_results[name] = result
        summary_metrics[name] = performance_summary(result)

        excess_df = build_factor_excess_table(result["portfolio_return"], benchmark["return"])
        factor_excess[config["factor"].upper()] = excess_df

        phase_metrics, _ = compute_phase_performance(result["portfolio_return"], phase_labels)
        phase_cagr_dict[name] = phase_metrics["CAGR"] if not phase_metrics.empty else pd.Series(dtype=float)

    print("Generating charts and tables...")
    overview_table = build_strategy_overview(configs)
    performance_table = build_performance_table(summary_metrics)
    phase_table = build_phase_performance_table({phase: {name: phase_cagr_dict[name].get(phase, float("nan")) for name in phase_cagr_dict} for phase in set().union(*[set(v.index) for v in phase_cagr_dict.values()])})
    if phase_table.empty:
        phase_table = pd.DataFrame(columns=[config["name"] for config in configs])

    equity_curves = {name: res["cum_return"] for name, res in portfolio_results.items()}
    charts = generate_report_charts(
        output_dir=output_dir,
        strategy_cum_returns=equity_curves,
        benchmark=benchmark,
        phase_cagr=phase_table,
        factor_excess=factor_excess,
    )

    overview_text = (
        "이 보고서는 KOSPI 우량주 기반 유니버스와 필터링 조건을 사용한 두 가지 전략을 비교합니다. "
        "각 전략은 섹터 비중을 추종하는 포트폴리오 구성, 최적화 방식, 거래비용 및 리밸런싱 주기를 포함합니다."
    )

    assembler = ReportAssembler(output_dir)
    markdown_path = assembler.write_markdown(
        path=Path(f"{args.report_name}.md"),
        title="Quant Strategy Report",
        overview=overview_text,
        tables={
            "Strategy Overview": overview_table,
            "Overall Performance": performance_table,
            "Phase Performance (CAGR)": phase_table,
        },
        images=charts,
    )

    print(f"Markdown report written to: {markdown_path}")
    docx_path = assembler.write_docx(
        path=Path(f"{args.report_name}.docx"),
        title="Quant Strategy Report",
        overview=overview_text,
        tables={
            "Strategy Overview": overview_table,
            "Overall Performance": performance_table,
            "Phase Performance (CAGR)": phase_table,
        },
        images=charts,
    )
    if docx_path:
        print(f"DOCX report written to: {docx_path}")
    else:
        print("python-docx not installed; skipping DOCX output.")


if __name__ == "__main__":
    main()
