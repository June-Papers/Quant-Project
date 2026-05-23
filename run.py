"""Simple factor backtest runner with structured report generation."""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from quant.data import DataLoader
from quant.backtester import Backtester
from quant.analytics import performance_summary
from quant.universe import filter_signal
import quant.factor as factor_module
from quant.benchmark import build_market_benchmark
from quant.report import (
    ReportAssembler,
    generate_report_charts,
    build_strategy_overview,
    build_factor_excess_table,
    compute_monthly_returns,
    compute_phase_labels,
    compute_phase_summary,
    build_comparison_performance_table,
    build_phase_comparison_table,
    build_annual_return_table,
    build_report_overview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured quant backtest runner and report generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--factor", type=str, default="pbr", choices=["pbr", "per", "roe"], help="Factor name")
    parser.add_argument("--group", type=str, default="Sector", choices=["Market", "Sector"], help="Neutralization group")
    parser.add_argument("--direction", type=str, default="low", choices=["low", "high"], help="Signal direction")
    parser.add_argument("--weighting", type=str, default="rank", choices=["equal", "rank", "score"], help="Weighting scheme")
    parser.add_argument("--allocation", type=str, default="signal", choices=["signal", "equal_weight", "risk_parity", "markowitz"], help="Portfolio allocation method")
    parser.add_argument("--top-n", type=int, default=None, help="Top N stocks per rebalance (if None, use top_pct)")
    parser.add_argument("--top-pct", type=float, default=0.2, help="Top percentage for rank weighting (0~1)")
    parser.add_argument("--max-weight", type=float, default=0.1, help="Maximum stock weight")
    parser.add_argument("--reb-freq", type=str, default="M", choices=["D", "W", "M", "Q", "Y"], help="Rebalance frequency")
    parser.add_argument("--transaction-cost", type=float, default=0.0025, help="Transaction cost")
    parser.add_argument("--universe-cap-threshold", type=float, default=100000.0, help="Minimum market cap in 백만원")
    parser.add_argument("--universe-turnover-quantile", type=float, default=0.05, help="Exclude lowest liquidity quantile")
    parser.add_argument("--universe-skip-halted", action="store_true", help="Do not exclude halted stocks on rebalance dates.")
    parser.add_argument("--universe-disable", action="store_true", help="Disable universe filtering entirely.")
    parser.add_argument("--data-path", type=str, default="../data", help="Path to source parquet data")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for report files")
    parser.add_argument("--report-name", type=str, default="strategy_report", help="Base report filename")
    parser.add_argument("--debug", action="store_true", help="Print debug info for first rebalance date")
    return parser.parse_args()


def load_data(data_path: str) -> Dict[str, pd.DataFrame]:
    dl = DataLoader(data_path=data_path)
    return {
        "close": dl.load_close(),
        "cap": dl.load_cap(),
        "sector": dl.load_sector(),
        "volume": dl.load_volume(),
        "shares_outstanding": dl.load_shares_outstanding(),
        "halted": dl.load_halted(),
        "bs": dl.load_bs(),
        "pl": dl.load_pl(),
        "data_loader": dl,
    }


def build_strategy_config(args: argparse.Namespace) -> List[Dict]:
    universe_description = (
        f"KOSPI universe filtered by market cap >= {args.universe_cap_threshold:.0f}백만원 "
        f"and excluding lowest {args.universe_turnover_quantile * 100:.1f}% liquidity names."
    )
    return [
        {
            "name": "Strategy",
            "factor": args.factor,
            "group": args.group,
            "direction": args.direction,
            "weighting": args.weighting,
            "allocation": args.allocation,
            "top_n": args.top_n,
            "top_pct": args.top_pct,
            "max_weight": args.max_weight,
            "reb_freq": args.reb_freq,
            "transaction_cost": args.transaction_cost,
            "universe_description": universe_description,
            "report_description": (
                "KOSPI 섹터 비중을 추종하면서 선택된 팩터를 기반으로 포트폴리오를 구성합니다."
            ),
        }
    ]


def build_factor(dl: DataLoader, config: Dict, close: pd.DataFrame, bs: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    factor_name = config["factor"].lower()
    if factor_name == "pbr":
        return factor_module.build_pbr(dl, close, bs)
    if factor_name == "per":
        return factor_module.build_per(dl, close, pl)
    if factor_name == "roe":
        return factor_module.build_roe(dl, close, bs, pl)
    raise ValueError(f"Unsupported factor: {config['factor']}")


def apply_universe_filter(signal: pd.DataFrame, args: argparse.Namespace, close: pd.DataFrame, cap: pd.DataFrame, volume: pd.DataFrame, shares_outstanding: pd.DataFrame, halted: pd.DataFrame) -> pd.DataFrame:
    if args.universe_disable:
        return signal
    return filter_signal(
        signal=signal,
        close=close,
        cap=cap,
        volume=volume,
        shares_outstanding=shares_outstanding,
        halted=halted,
        cap_threshold=args.universe_cap_threshold,
        turnover_quantile=args.universe_turnover_quantile,
        exclude_halted=not args.universe_skip_halted,
        reb_freq=args.reb_freq,
    )


def build_benchmark_result(benchmark_return: pd.Series, benchmark_cum: pd.Series) -> Dict[str, pd.Series]:
    return {
        "portfolio_return": benchmark_return,
        "cum_return": benchmark_cum,
        "turnover": pd.Series(0.0, index=benchmark_return.index),
    }


def save_summary_outputs(output_dir: Path, result: Dict[str, pd.Series], summary: Dict[str, float]) -> None:
    daily_result = pd.DataFrame({
        "portfolio_return": result["portfolio_return"],
        "cum_return": result["cum_return"],
        "turnover": result["turnover"],
    })
    daily_result.to_csv(output_dir / "daily_backtest_result.csv")

    summary_df = pd.DataFrame(summary, index=[0]).T.reset_index()
    summary_df.columns = ["Metric", "Value"]
    summary_df.to_csv(output_dir / "performance_summary.csv", index=False)

    if "weights" in result:
        result["weights"].to_csv(output_dir / "daily_weights.csv")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(args.data_path)
    close = data["close"]
    cap = data["cap"]
    sector = data["sector"]
    volume = data["volume"]
    shares_outstanding = data["shares_outstanding"]
    halted = data["halted"]
    bs = data["bs"]
    pl = data["pl"]
    dl = data["data_loader"]

    print("Building benchmark...")
    benchmark = build_market_benchmark(close, cap)

    configs = build_strategy_config(args)
    config = configs[0]

    print(f"Building factor: {config['factor']}")
    factor = build_factor(dl, config, close, bs, pl)
    factor = apply_universe_filter(factor, args, close, cap, volume, shares_outstanding, halted)

    if args.debug:
        print(f"Factor shape: {factor.shape}")

    backtester = Backtester(close=close, cap=cap, sector=sector)
    result = backtester.backtest(
        signal=factor,
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

    benchmark_result = build_benchmark_result(benchmark["return"], benchmark["cum_return"])
    benchmark_summary = performance_summary(benchmark_result)
    summary = performance_summary(result)
    save_summary_outputs(output_dir, result, summary)

    strategy_name = config["name"]
    overview_df = build_strategy_overview(configs).T
    overview_df.index.name = "Metric"
    performance_df = build_comparison_performance_table(summary, benchmark_summary)

    strategy_monthly = compute_monthly_returns(result["portfolio_return"])
    benchmark_monthly = compute_monthly_returns(benchmark["return"])
    phase_labels = compute_phase_labels(benchmark["cum_return"])
    strategy_phase_summary, _ = compute_phase_summary(strategy_monthly, phase_labels)
    benchmark_phase_summary, _ = compute_phase_summary(benchmark_monthly, phase_labels)
    phase_summary_df = build_phase_comparison_table(strategy_phase_summary, benchmark_phase_summary)
    phase_cagr = pd.DataFrame({
        "Strategy": strategy_phase_summary["CAGR"],
        "Benchmark": benchmark_phase_summary["CAGR"],
    }).dropna()

    annual_returns_df = build_annual_return_table(result["portfolio_return"], benchmark["return"])
    excess_df = build_factor_excess_table(result["portfolio_return"], benchmark["return"])

    chart_paths = generate_report_charts(
        output_dir=output_dir,
        strategy_cum_returns={strategy_name: result["cum_return"]},
        benchmark=benchmark,
        phase_cagr=phase_cagr,
        factor_excess={config["factor"].upper(): excess_df},
        strategy_cum_return=result["cum_return"],
        benchmark_cum_return=benchmark["cum_return"],
        annual_excess_returns=annual_returns_df[["Excess"]],
    )

    report_overview = build_report_overview(config, benchmark["cum_return"], summary, benchmark_summary)

    assembler = ReportAssembler(output_dir)
    markdown_path = assembler.write_markdown(
        path=Path(f"{args.report_name}.md"),
        title="Quant Strategy Report",
        overview=report_overview,
        tables={
            "Strategy Overview": overview_df,
            "Overall Performance": performance_df,
            "Phase Summary": phase_summary_df,
            "Annual Returns": annual_returns_df,
        },
        images=chart_paths,
    )

    print(f"Report generated: {markdown_path}")
    docx_path = assembler.write_docx(
        path=Path(f"{args.report_name}.docx"),
        title="Quant Strategy Report",
        overview=report_overview,
        tables={
            "Strategy Overview": overview_df,
            "Overall Performance": performance_df,
            "Phase Summary": phase_summary_df,
            "Annual Returns": annual_returns_df,
        },
        images=chart_paths,
    )

    if docx_path is not None:
        print(f"DOCX report generated: {docx_path}")
    else:
        print("python-docx not installed; DOCX report skipped.")

    print("Outputs saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
