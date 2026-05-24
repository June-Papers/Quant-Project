from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _select_excel_engine() -> str:
    if find_spec("openpyxl") is not None:
        return "openpyxl"
    if find_spec("xlsxwriter") is not None:
        return "xlsxwriter"
    raise ImportError("No Excel writer engine found. Install openpyxl or xlsxwriter.")


def read_excel_template(template_path: Path) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(template_path, sheet_name=None)


def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    else:
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _safe_sheet_name(name: str, existing: Iterable[str]) -> str:
    invalid = '[]:*?/\\'
    safe = "".join("_" if ch in invalid else ch for ch in str(name)).strip() or "Sheet"
    safe = safe[:31]
    used = set(existing)
    if safe not in used:
        return safe

    base = safe[:28]
    idx = 1
    while True:
        candidate = f"{base}_{idx}"[:31]
        if candidate not in used:
            return candidate
        idx += 1


def _strategy_label(config: Optional[Dict], fallback: str) -> str:
    if not config:
        return fallback
    factor = str(config.get("factor", "")).strip()
    return factor.upper() if factor else str(config.get("name", fallback))


def _monthly_returns(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return (1 + r).resample("M").prod() - 1


def _annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    return r.std() * np.sqrt(periods_per_year) if len(r) > 1 else np.nan


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(periods_per_year)


def _max_drawdown(cum_return: pd.Series) -> float:
    cum = cum_return.dropna()
    if cum.empty:
        return np.nan
    return (cum / cum.cummax() - 1).min()


def _performance_metrics(result: Dict[str, pd.Series]) -> Dict[str, float]:
    returns = result["portfolio_return"].dropna()
    cum = result.get("cum_return")
    if cum is None:
        cum = (1 + returns.fillna(0)).cumprod()
    cum = cum.dropna()
    monthly = _monthly_returns(returns)
    positive_months = monthly[monthly > 0]
    negative_months = monthly[monthly < 0]
    years = (cum.index[-1] - cum.index[0]).days / 365.25 if len(cum) > 1 else np.nan

    return {
        "CAGR": cum.iloc[-1] ** (1 / years) - 1 if years and years > 0 else np.nan,
        "누적수익률": cum.iloc[-1] - 1 if not cum.empty else np.nan,
        "연간변동성": _annualized_volatility(returns),
        "Sharpe Ratio": _sharpe(returns),
        "MDD": _max_drawdown(cum),
        "양수월 비율(수익난 달 / 전체달)": len(positive_months) / len(monthly) if len(monthly) else np.nan,
        "양수월 평균 수익률": positive_months.mean() if len(positive_months) else np.nan,
        "음수월 평균 수익률": negative_months.mean() if len(negative_months) else np.nan,
    }


def _build_performance_sheet(
    strategy_results: Dict[str, Dict[str, pd.Series]],
    strategy_configs: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    metric_order = [
        "CAGR",
        "누적수익률",
        "연간변동성",
        "Sharpe Ratio",
        "MDD",
        "양수월 비율(수익난 달 / 전체달)",
        "양수월 평균 수익률",
        "음수월 평균 수익률",
    ]
    notes = {
        "CAGR": "연율화 복리수익률",
        "누적수익률": "백테스트 기간 전체 수익률",
        "연간변동성": "일간 수익률 기준 연율화",
        "Sharpe Ratio": "무위험수익률 0 가정",
        "MDD": "누적수익률 기준 최대 낙폭",
        "양수월 비율(수익난 달 / 전체달)": "월간 수익률이 0보다 큰 달 비율",
        "양수월 평균 수익률": "양수 월간 수익률 평균",
        "음수월 평균 수익률": "음수 월간 수익률 평균",
    }

    rows = []
    metrics_by_strategy = {
        _strategy_label((strategy_configs or {}).get(name), name): _performance_metrics(result)
        for name, result in strategy_results.items()
    }
    for metric in metric_order:
        row = {"지표": metric}
        for label, metrics in metrics_by_strategy.items():
            row[label] = metrics.get(metric, np.nan)
        row["비고"] = notes.get(metric, "")
        rows.append(row)
    return pd.DataFrame(rows)


def _sector_weight_history(weights: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    weights_idx = _to_date_index(weights)
    sector_idx = _to_date_index(sector).reindex(weights_idx.index).ffill()
    rows = []
    for dt, weight_row in weights_idx.iterrows():
        positive = weight_row[weight_row > 0]
        if positive.empty:
            continue
        sector_row = _clean_sector_labels(sector_idx.loc[dt].reindex(positive.index))
        row = positive.groupby(sector_row).sum()
        row.name = dt
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    history = pd.DataFrame(rows).fillna(0).sort_index()
    history.index.name = "날짜"
    return history.resample("M").last().dropna(how="all")


def _clean_sector_labels(labels: pd.Series) -> pd.Series:
    cleaned = labels.astype(object).where(~labels.isna(), "미분류")
    return cleaned.replace({"nan": "미분류", "NaN": "미분류", "": "미분류"})


def _included_sector_count(weights: pd.DataFrame, sector: pd.DataFrame) -> pd.Series:
    history = _sector_weight_history(weights, sector)
    if history.empty:
        return pd.Series(dtype=float)
    return (history > 0).sum(axis=1).rename("편입업종수")


def _build_monthly_sheet(
    strategy_results: Dict[str, Dict[str, pd.Series]],
    benchmark_return: pd.Series,
    sector: pd.DataFrame,
    strategy_configs: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    parts = []
    for name, result in strategy_results.items():
        label = _strategy_label((strategy_configs or {}).get(name), name)
        parts.append(_monthly_returns(result["portfolio_return"]).rename(label))
    parts.append(_monthly_returns(benchmark_return).rename("KOSPI"))
    monthly = pd.concat(parts, axis=1).sort_index()

    count_parts = []
    for name, result in strategy_results.items():
        label = _strategy_label((strategy_configs or {}).get(name), name)
        count_name = "편입업종수" if len(strategy_results) == 1 else f"{label} 편입업종수"
        count_parts.append(_included_sector_count(result.get("weights", pd.DataFrame()), sector).rename(count_name))
    if count_parts:
        monthly = monthly.join(pd.concat(count_parts, axis=1), how="left")
    monthly.index.name = "날짜"
    return monthly.reset_index()


def _load_mapping_table(data_path: Path) -> pd.Series:
    mapping_path = data_path / "mapping_table.csv"
    if not mapping_path.exists():
        return pd.Series(dtype=object)
    try:
        mapping = pd.read_csv(mapping_path)
    except UnicodeDecodeError:
        mapping = pd.read_csv(mapping_path, encoding="cp949")
    if {"코드", "코드명"}.issubset(mapping.columns):
        return mapping.set_index("코드")["코드명"]
    return pd.Series(dtype=object)


def _period_return(close: pd.DataFrame, date: pd.Timestamp, stocks: pd.Index, periods: int) -> pd.Series:
    close_idx = _to_date_index(close).reindex(columns=stocks)
    eligible = close_idx.loc[:date]
    if len(eligible) <= periods:
        return pd.Series(np.nan, index=stocks)
    current = eligible.iloc[-1]
    previous = eligible.iloc[-periods - 1]
    return current / previous - 1


def _current_portfolio_sheets(
    result: Dict[str, pd.Series],
    close: pd.DataFrame,
    cap: pd.DataFrame,
    sector: pd.DataFrame,
    stock_names: pd.Series,
) -> Dict[str, pd.DataFrame]:
    weights = result.get("weights", pd.DataFrame())
    if weights.empty:
        return {
            "stocks": pd.DataFrame(),
            "sectors": pd.DataFrame(),
            "sector_history": pd.DataFrame(),
        }

    weights_idx = _to_date_index(weights)
    nonzero = weights_idx[(weights_idx > 0).any(axis=1)]
    if nonzero.empty:
        current = weights_idx.iloc[-1]
        current_date = weights_idx.index[-1]
    else:
        current = nonzero.iloc[-1]
        current_date = nonzero.index[-1]
    current = current[current > 0].sort_values(ascending=False)
    stocks = pd.Index(current.index)

    sector_idx = _to_date_index(sector).reindex(weights_idx.index).ffill()
    current_sector = _clean_sector_labels(sector_idx.loc[current_date].reindex(stocks))
    sector_weights = current.groupby(current_sector).sum()
    within_sector = current / current_sector.map(sector_weights)
    cap_idx = _to_date_index(cap).reindex(columns=stocks)
    latest_cap = cap_idx.loc[:current_date].iloc[-1].reindex(stocks) if not cap_idx.loc[:current_date].empty else pd.Series(np.nan, index=stocks)

    stock_df = pd.DataFrame({
        "종목코드": stocks,
        "종목명": stock_names.reindex(stocks).fillna("").values,
        "업종": current_sector.values,
        "업종비중": current_sector.map(sector_weights).values,
        "업종내비중": within_sector.reindex(stocks).values,
        "포트폴리오비중": current.reindex(stocks).values,
        "1주 수익률": _period_return(close, current_date, stocks, 5).values,
        "1개월 수익률": _period_return(close, current_date, stocks, 21).values,
        "3개월 수익률": _period_return(close, current_date, stocks, 63).values,
        "1일 수익률": _period_return(close, current_date, stocks, 1).values,
        "시가총액": latest_cap.values,
    })

    sector_df = (
        stock_df.groupby("업종", dropna=False)
        .agg(
            업종비중=("포트폴리오비중", "sum"),
            해당업종편입종목수=("종목코드", "count"),
            편입종목=("종목명", lambda x: ", ".join([str(v) for v in x if str(v)])),
        )
        .reset_index()
        .sort_values("업종비중", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "stocks": stock_df,
        "sectors": sector_df,
        "sector_history": _sector_weight_history(weights_idx, sector),
    }


def build_backtest_excel_tables(
    strategy_results: Dict[str, Dict[str, pd.Series]],
    benchmark_return: pd.Series,
    close: pd.DataFrame,
    cap: pd.DataFrame,
    sector: pd.DataFrame,
    data_path: Path,
    strategy_configs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    used_sheet_names: List[str] = []

    def add_sheet(name: str, df: pd.DataFrame) -> None:
        safe_name = _safe_sheet_name(name, used_sheet_names)
        used_sheet_names.append(safe_name)
        tables[safe_name] = df

    add_sheet("백테스트 성과", _build_performance_sheet(strategy_results, strategy_configs))
    add_sheet("월간 수익률 시계열", _build_monthly_sheet(strategy_results, benchmark_return, sector, strategy_configs))

    stock_names = _load_mapping_table(Path(data_path))
    for name, result in strategy_results.items():
        label = _strategy_label((strategy_configs or {}).get(name), name)
        current = _current_portfolio_sheets(result, close, cap, sector, stock_names)
        add_sheet(f"{label} 현재 포트폴리오 종목", current["stocks"])
        add_sheet(f"{label} 현재 포트폴리오 업종", current["sectors"])
        add_sheet(f"{label} 업종 히스토리", current["sector_history"])

    return tables


def write_excel_report(
    output_path: Path,
    tables: Dict[str, pd.DataFrame],
    template_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _select_excel_engine()
    with pd.ExcelWriter(output_path, engine=engine) as writer:
        if template_data is not None:
            for sheet_name, sheet_df in template_data.items():
                sheet_name_safe = sheet_name[:31]
                sheet_df.to_excel(writer, sheet_name=sheet_name_safe, index=False)

        for section, df in tables.items():
            sheet_name = section if len(section) <= 31 else section[:31]
            if template_data is not None and sheet_name in template_data:
                sheet_name = f"{sheet_name}_report"[:31]
            write_index = not isinstance(df.index, pd.RangeIndex) and df.index.name is not None
            df.to_excel(writer, sheet_name=sheet_name, index=write_index)

    return output_path
