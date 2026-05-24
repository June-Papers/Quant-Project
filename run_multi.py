"""Multi-strategy backtest runner with KOSPI benchmark comparison."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

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
    build_performance_table,
    build_phase_performance_table,
    build_factor_excess_table,
    build_annual_return_table,
    build_backtest_excel_tables,
    read_excel_template,
    write_excel_report,
    compute_monthly_returns,
    compute_phase_labels,
    compute_phase_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple strategy backtests and compare them against the KOSPI benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        help=(
            "Strategy definition as NAME|FACTOR|GROUP|DIRECTION|WEIGHTING|ALLOCATION|TOP_PCT|MAX_WEIGHT|REB_FREQ|TRANSACTION_COST "
            "or FACTOR|GROUP|DIRECTION|WEIGHTING|ALLOCATION|TOP_PCT|MAX_WEIGHT|REB_FREQ|TRANSACTION_COST if name is omitted. "
            "Example: PBR_STRAT|pbr|Sector|low|rank|markowitz|0.2|0.15|M|0.001"
        ),
    )
    parser.add_argument("--universe-cap-threshold", type=float, default=100000.0, help="Minimum market cap in 백만원")
    parser.add_argument("--universe-turnover-quantile", type=float, default=0.05, help="Exclude lowest liquidity quantile")
    parser.add_argument("--universe-skip-halted", action="store_true", help="Do not exclude halted stocks on rebalance dates.")
    parser.add_argument("--universe-disable", action="store_true", help="Disable universe filtering entirely.")
    parser.add_argument("--reb-freq", type=str, default="M", choices=["D", "W", "M", "Q", "Y"], help="Universe rebalance frequency for filtering")
    parser.add_argument("--data-path", type=str, default="../data", help="Path to source parquet data")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for report files")
    parser.add_argument("--report-name", type=str, default="multi_strategy_report", help="Base report filename")
    parser.add_argument("--excel-template", type=str, default=None, help="Optional Excel template file to read when generating the workbook")
    parser.add_argument("--excel-output", type=str, default=None, help="Optional Excel output file path")
    parser.add_argument("--debug", action="store_true", help="Print debug info for each strategy")
    return parser.parse_args()


def parse_strategy_spec(spec: str, index: int) -> Dict:
    tokens = [token.strip() for token in spec.split("|") if token.strip() != ""]
    if len(tokens) == 10:
        name, factor, group, direction, weighting, allocation, top_pct, max_weight, reb_freq, transaction_cost = tokens
    elif len(tokens) == 9:
        name = f"Strategy {index + 1}"
        factor, group, direction, weighting, allocation, top_pct, max_weight, reb_freq, transaction_cost = tokens
    else:
        raise ValueError(
            "Strategy spec must be NAME|FACTOR|GROUP|DIRECTION|WEIGHTING|ALLOCATION|TOP_PCT|MAX_WEIGHT|REB_FREQ|TRANSACTION_COST "
            "or FACTOR|GROUP|DIRECTION|WEIGHTING|ALLOCATION|TOP_PCT|MAX_WEIGHT|REB_FREQ|TRANSACTION_COST"
        )

    return {
        "name": name,
        "factor": factor.lower(),
        "group": group,
        "direction": direction,
        "weighting": weighting,
        "allocation": allocation,
        "top_n": None,
        "top_pct": float(top_pct),
        "max_weight": float(max_weight),
        "reb_freq": reb_freq,
        "transaction_cost": float(transaction_cost),
        "universe_description": "KOSPI universe filtered by market cap and liquidity.",
        "report_description": "Multi-strategy comparison against the KOSPI benchmark.",
    }


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


def build_factor(dl: DataLoader, config: Dict, close: pd.DataFrame, bs: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    factor_name = config["factor"].lower()
    if factor_name == "pbr":
        return factor_module.build_pbr(dl, close, bs)
    if factor_name == "per":
        return factor_module.build_per(dl, close, pl)
    if factor_name == "roe":
        return factor_module.build_roe(dl, close, bs, pl)
    raise ValueError(f"Unsupported factor: {config['factor']}")


def apply_universe_filter(
    signal: pd.DataFrame,
    args: argparse.Namespace,
    close: pd.DataFrame,
    cap: pd.DataFrame,
    volume: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
    halted: pd.DataFrame,
) -> pd.DataFrame:
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


def build_overview_text(configs: List[Dict]) -> str:
    description_lines = [
        "## 요약\n\n",
        "이 리포트는 KOSPI 유니버스를 기반으로 한 여러 전략을 벤치마크와 비교합니다.\n\n",
        "전략 엔진은 다음과 같은 설계를 따릅니다:\n",
        "1) group == Sector: 시장의 섹터 비중을 추종하며, 섹터 내에서 후보 종목 비중을 결정합니다.\n",
        "2) group == Market: 섹터 구분 없이 전체 시장에서 신호에 따라 종목을 선별합니다.\n",
        "3) 각 전략은 top_pct를 사용해 1차 후보군을 구성하고, allocation 방식에 따라 비중을 최적화합니다.\n",
        "4) 최종 포트폴리오는 최대 종목 비중(max_weight)을 반영해 과도한 집중도를 제한합니다.\n\n",
        "전략별 구성:\n",
    ]

    for config in configs:
        label = "섹터 내" if config["group"] == "Sector" else "섹터 구분 없이"
        if config["allocation"] == "signal":
            allocation_text = f"{config['weighting']} 방식으로 편입합니다."
        else:
            allocation_text = f"{config['allocation']} 방식으로 비중을 최적화합니다."

        description_lines.append(
            f"- {config['name']}: {label} {config['direction']} {config['factor']} 종목을 top_pct={config['top_pct']:.0%}로 1차 후보군으로 설정하고, {allocation_text} "
            f"최대 종목 비중은 {config['max_weight']:.2%}입니다.\n"
        )

    return "".join(description_lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_configs = [parse_strategy_spec(spec, idx) for idx, spec in enumerate(args.strategy)]

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
    benchmark_summary = performance_summary({
        "portfolio_return": benchmark["return"],
        "cum_return": benchmark["cum_return"],
        "turnover": pd.Series(0.0, index=benchmark["return"].index),
    })

    backtester = Backtester(close=close, cap=cap, sector=sector)

    portfolio_results: Dict[str, Dict[str, pd.Series]] = {}
    summary_metrics: Dict[str, Dict[str, float]] = {}
    factor_excess: Dict[str, pd.DataFrame] = {}
    annual_excess_returns: Dict[str, pd.DataFrame] = {}
    phase_cagr_dict: Dict[str, pd.Series] = {}
    config_by_name: Dict[str, Dict] = {}

    phase_labels = compute_phase_labels(benchmark["cum_return"])

    for config in strategy_configs:
        print(f"Running strategy: {config['name']}")
        factor = build_factor(dl, config, close, bs, pl)
        factor = apply_universe_filter(factor, args, close, cap, volume, shares_outstanding, halted)

        if args.debug:
            print(f"  factor shape: {factor.shape}")

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

        portfolio_results[config["name"]] = result
        config_by_name[config["name"]] = config
        summary_metrics[config["name"]] = performance_summary(result)
        factor_excess[config["name"]] = build_factor_excess_table(result["portfolio_return"], benchmark["return"])
        annual_excess_returns[config["name"]] = build_annual_return_table(result["portfolio_return"], benchmark["return"])[["Excess"]]

        strategy_monthly = compute_monthly_returns(result["portfolio_return"])
        strategy_phase_summary, _ = compute_phase_summary(strategy_monthly, phase_labels)
        phase_cagr_dict[config["name"]] = strategy_phase_summary["CAGR"] if not strategy_phase_summary.empty else pd.Series(dtype=float)

    full_performance_table = build_performance_table({**summary_metrics, "Benchmark": benchmark_summary})

    benchmark_phase_summary, _ = compute_phase_summary(compute_monthly_returns(benchmark["return"]), phase_labels)
    phase_cagr_dict["Benchmark"] = benchmark_phase_summary["CAGR"] if not benchmark_phase_summary.empty else pd.Series(dtype=float)

    phase_table = build_phase_performance_table({
        phase: {name: phase_cagr_dict[name].get(phase, float("nan")) for name in phase_cagr_dict}
        for phase in set().union(*[set(series.index) for series in phase_cagr_dict.values()])
    })
    if phase_table.empty:
        phase_table = pd.DataFrame(columns=list(phase_cagr_dict.keys()))

    overview_table = build_strategy_overview(strategy_configs)
    overview_text = build_overview_text(strategy_configs)

    equity_curves = {name: res["cum_return"] for name, res in portfolio_results.items()}
    chart_paths = generate_report_charts(
        output_dir=output_dir,
        strategy_cum_returns={**equity_curves},
        benchmark=benchmark,
        phase_cagr=phase_table,
        factor_excess=factor_excess,
        strategy_cum_return=next(iter(equity_curves.values())) if equity_curves else None,
        benchmark_cum_return=benchmark["cum_return"],
        annual_excess_returns=annual_excess_returns,
    )

    excel_output = Path(args.excel_output) if args.excel_output else output_dir / f"{args.report_name}.xlsx"
    template_data = None
    if args.excel_template:
        template_data = read_excel_template(Path(args.excel_template))
    write_excel_report(
        excel_output,
        tables=build_backtest_excel_tables(
            strategy_results=portfolio_results,
            benchmark_return=benchmark["return"],
            close=close,
            cap=cap,
            sector=sector,
            data_path=Path(args.data_path),
            strategy_configs=config_by_name,
        ),
        template_data=template_data,
    )
    print(f"Excel report generated: {excel_output}")

    assembler = ReportAssembler(output_dir)
    markdown_path = assembler.write_markdown(
        path=Path(f"{args.report_name}.md"),
        title="Multi Strategy Quant Report",
        overview=overview_text,
        tables={
            "Strategy Overview": overview_table,
            "Performance Summary": full_performance_table,
            "Phase CAGR Comparison": phase_table,
        },
        images=chart_paths,
    )

    print(f"Report generated: {markdown_path}")

    docx_path = assembler.write_docx(
        path=Path(f"{args.report_name}.docx"),
        title="Multi Strategy Quant Report",
        overview=overview_text,
        tables={
            "Strategy Overview": overview_table,
            "Performance Summary": full_performance_table,
            "Phase CAGR Comparison": phase_table,
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
