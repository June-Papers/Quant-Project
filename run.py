"""Simple factor backtest runner."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from quant.data import DataLoader
from quant.backtester import Backtester
from quant.analytics import performance_summary
from quant.universe import filter_signal

import quant.factor as factor_module



# ============================================
# Argument Parser
# ============================================
parser = argparse.ArgumentParser(
    description="Simple Quant Backtest Runner",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--factor",
    type=str,
    default="pbr",
    choices=["pbr", "per", "roe"],
    help="Factor name"
)

parser.add_argument(
    "--group",
    type=str,
    default="Sector",
    choices=["Market", "Sector"],
    help="Neutralization group"
)

parser.add_argument(
    "--direction",
    type=str,
    default="low",
    choices=["low", "high"],
    help="Signal direction"
)

parser.add_argument(
    "--weighting",
    type=str,
    default="rank",
    choices=["equal", "rank", "score"],
    help="Weighting scheme"
)

parser.add_argument(
    "--allocation",
    type=str,
    default="signal",
    choices=["signal", "equal_weight", "risk_parity", "markowitz"],
    help="Portfolio allocation method after selection"
)

parser.add_argument(
    "--top-n",
    type=int,
    default=None,
    help="Top N stocks per rebalance (if None, use top_pct)"
)

parser.add_argument(
    "--top-pct",
    type=float,
    default=0.2,
    help="Top percentage for rank weighting (0~1)"
)

parser.add_argument(
    "--max-weight",
    type=float,
    default=0.1,
    help="Maximum stock weight"
)

parser.add_argument(
    "--reb-freq",
    type=str,
    default="M",
    choices=["D", "W", "M", "Q", "Y"],
    help="Rebalance frequency"
)

parser.add_argument(
    "--transaction-cost",
    type=float,
    default=0.0025,
    help="Transaction cost"
)

parser.add_argument(
    "--universe-cap-threshold",
    type=float,
    default=100000.0,
    help="Minimum market cap in 백만원 to remain in universe (default 100000 = 1000억원)."
)

parser.add_argument(
    "--universe-turnover-quantile",
    type=float,
    default=0.05,
    help="Exclude stocks below this prior-month turnover quantile (default 0.05)."
)

parser.add_argument(
    "--universe-skip-halted",
    action="store_true",
    help="Do not exclude halted stocks on rebalance dates."
)

parser.add_argument(
    "--universe-disable",
    action="store_true",
    help="Disable universe filtering entirely."
)

parser.add_argument(
    "--debug",
    action="store_true",
    help="Print debug info for first rebalance date"
)

args = parser.parse_args()


# ============================================
# Output Directory
# ============================================
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)


# ============================================
# Load Data
# ============================================
print("Loading data...")

dl = DataLoader(data_path="../data")

close = dl.load_close()
cap = dl.load_cap()
sector = dl.load_sector()
volume = dl.load_volume()
shares_outstanding = dl.load_shares_outstanding()
halted = dl.load_halted()
bs = dl.load_bs()
pl = dl.load_pl()

print("Data loaded.")

# Remove outliers 임시용, extreme outlier 종목이 존재하여 제거 (예: A008080)
close = close.drop(columns='A008080', errors='ignore')
cap = cap.drop(columns='A008080', errors='ignore')
sector = sector.drop(columns='A008080', errors='ignore')


# ============================================
# Build Factor
# ============================================
print(f"Building factor: {args.factor}")

import quant.factor as factor_module
pl = dl.load_pl()

if args.factor.lower() == "pbr":
    factor = factor_module.build_pbr(dl, close, bs)
elif args.factor.lower() == "per":
    factor = factor_module.build_per(dl, close, pl)
elif args.factor.lower() == "roe":
    factor = factor_module.build_roe(dl, close, bs, pl)
else:
    raise ValueError(f"Unknown factor: {args.factor}. Use 'pbr', 'per', 'roe'")

print(f"✓ Factor shape: {factor.shape}")

if not args.universe_disable:
    factor = filter_signal(
        signal=factor,
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
    print("✓ Universe filtering applied")
else:
    print("✓ Universe filtering disabled")

if args.debug:
    # prepare index-aligned factor
    fac_df = factor.set_index('date') if 'date' in factor.columns else factor.copy()
    # get rebalance dates
    from quant.utils import get_rebalance_dates
    reb_dates = get_rebalance_dates(fac_df.index, args.reb_freq)
    if len(reb_dates) > 0:
        # find first rebalance date that has any non-null signal
        dt = None
        for d in reb_dates:
            if d in fac_df.index and not fac_df.loc[d].dropna().empty:
                dt = d
                break

        if dt is None:
            print("[DEBUG] No non-empty rebalance date found for factor index.")
        else:
            print(f"\n[DEBUG] First non-empty rebalance date: {dt}")
            row = fac_df.loc[dt].dropna()
            print("[DEBUG] Signal (head):")
            print(row.head(10))

            # compute raw
            if args.direction == 'low':
                raw = -row
            else:
                raw = row

            print("[DEBUG] Raw (head):")
            print(raw.head(10))

            # score by z-score
            std = raw.std() if raw.std() != 0 else 1
            z = (raw - raw.mean()) / std
            print("[DEBUG] Z-score (head):")
            print(z.head(10).clip(lower=0))

            # rank score
            n = len(row)
            rank = row.rank(method='average')
            if args.direction == 'low':
                rank_score = (n - rank + 1) / n
            else:
                rank_score = rank / n
            cutoff = 1 - args.top_pct
            rank_cut = rank_score.where(rank_score >= cutoff, 0)
            print("[DEBUG] Rank score (head):")
            print(rank_cut.head(10))
    else:
        print("[DEBUG] No rebalance dates found for factor index.")


# ============================================
# Backtest
# ============================================
print("Running backtest...")

bt = Backtester(
    close=close,
    cap=cap,
    sector=sector
)

result = bt.backtest(
    signal=factor,
    group=args.group,
    direction=args.direction,
    weighting=args.weighting,
    top_n=args.top_n,
    top_pct=args.top_pct,
    max_weight=args.max_weight,
    reb_freq=args.reb_freq,
    portfolio_method=args.allocation,
    transaction_cost=args.transaction_cost
)

print("Backtest complete.")


# ============================================
# Performance Summary
# ============================================
summary = performance_summary(result)

print("\n=== Performance Summary ===")

for k, v in summary.items():
    print(f"{k}: {v:.4f}")


# ============================================
# Save Daily Backtest Result
# ============================================
daily_result = pd.DataFrame({
    "portfolio_return": result["portfolio_return"],
    "cum_return": result["cum_return"],
    "turnover": result["turnover"]
})

daily_result.to_csv(
    output_dir / "daily_backtest_result.csv"
)

print("✓ Daily result saved")


# ============================================
# Plot
# ============================================
print("Generating plots...")

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14, 10)
)

# --------------------------------------------
# 1. Cumulative Return
# --------------------------------------------
ax = axes[0]

ax.plot(
    result["cum_return"].index,
    result["cum_return"].values,
    linewidth=2
)

ax.set_title(
    f"{args.factor.upper()} ({args.group}) - Cumulative Return",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("Return")
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m")
)

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=6)
)

plt.setp(
    ax.xaxis.get_majorticklabels(),
    rotation=45
)


# --------------------------------------------
# 2. Daily Return
# --------------------------------------------
ax = axes[1]

daily_ret = result["portfolio_return"].fillna(0)

colors = [
    "green" if r > 0 else "red"
    for r in daily_ret
]

ax.bar(
    daily_ret.index,
    daily_ret.values,
    color=colors,
    width=1,
    alpha=0.6
)

ax.set_title(
    "Daily Portfolio Return",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("Daily Return")
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m")
)

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=6)
)

plt.setp(
    ax.xaxis.get_majorticklabels(),
    rotation=45
)


# --------------------------------------------
# 3. Turnover
# --------------------------------------------
ax = axes[2]

ax.plot(
    result["turnover"].index,
    result["turnover"].values,
    linewidth=1.5
)

ax.fill_between(
    result["turnover"].index,
    result["turnover"].values,
    alpha=0.3
)

ax.set_title(
    "Portfolio Turnover",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("Turnover")
ax.set_xlabel("Date")

ax.grid(True, alpha=0.3)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m")
)

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=6)
)

plt.setp(
    ax.xaxis.get_majorticklabels(),
    rotation=45
)

plt.tight_layout()

plot_path = output_dir / f"{args.factor}_{args.group}_{args.allocation}_{args.top_n or args.top_pct}.png"

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight"
)

print(f"✓ Plot saved: {plot_path}")


# ============================================
# Markdown Report
# ============================================
print("Generating markdown report...")

report_text = f"""
# Quant Backtest Report

## Strategy Information

| Item | Value |
|---|---|
| Factor | {args.factor} |
| Group Neutralization | {args.group} |
| Direction | {args.direction} |
| Weighting | {args.weighting} |
| Allocation Method | {args.allocation} |
| Max Weight | {args.max_weight} |
| Rebalance Frequency | {args.reb_freq} |
| Transaction Cost | {args.transaction_cost} |

---

## Performance Summary

| Metric | Value |
|---|---|
"""

for k, v in summary.items():

    report_text += f"| {k} | {v:.4f} |\n"

report_text += """

---

## Included Features

- Daily backtest
- Transaction cost applied
- Portfolio turnover calculation
- Cumulative return tracking
- Sector neutral portfolio construction

---

## Interpretation

This strategy constructs a factor portfolio using the selected signal
and applies periodic rebalancing.

The portfolio includes transaction costs and turnover-based rebalancing
effects, making the result closer to implementable performance.

The strategy performance should be interpreted carefully because
real market impact, execution latency, and liquidity constraints
are simplified.

---

## Limitations

- No liquidity filter
- No delisting handling
- No slippage model beyond fixed transaction cost
- No borrow cost for short positions
- Corporate actions may not be fully adjusted

---

## Future Improvements

- Add multi-factor support
- Add volatility scaling
- Add beta neutralization
- Add realistic slippage model
- Add long-short portfolio support
- Add benchmark comparison
"""

report_path = output_dir / "report.md"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print("✓ Markdown report saved")


# ============================================
# Save Summary CSV
# ============================================
summary_df = pd.DataFrame(
    summary.items(),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    output_dir / "performance_summary.csv",
    index=False
)



print("✓ Summary CSV saved")


# ============================================
# Save Daily Weights
# ============================================
if "weights" in result:

    result["weights"].to_csv(
        output_dir / "daily_weights.csv"
    )

    print("✓ Daily weights saved")


# ============================================
# Benchmark: KOSPI comparison
# ============================================
print("Generating KOSPI benchmark comparison...")

# Load index parquet (expects 'date' and '코스피' columns)
try:
    index_df = dl._read("index.parquet")
except Exception:
    import pandas as _pd
    from pathlib import Path as _P
    index_df = _pd.read_parquet(_P("../data") / "index.parquet")

index_df["date"] = pd.to_datetime(index_df["date"]) if "date" in index_df.columns else index_df.index
kospi = (
    index_df[["date", "코스피"]]
    .set_index("date")
    .sort_index()
)

# Align benchmark to daily backtest index
daily_index = pd.to_datetime(daily_result.index)
kospi_price = kospi["코스피"].reindex(daily_index).ffill()
kospi_ret = kospi_price.pct_change().fillna(0)
kospi_cum = (1 + kospi_ret).cumprod()

# Add benchmark series to daily_result for convenience
daily_compare = daily_result.copy()
daily_compare["kospi_price"] = kospi_price.values
daily_compare["kospi_return"] = kospi_ret.values
daily_compare["kospi_cum_return"] = kospi_cum.values

# Drawdown function
def compute_drawdown(cum_series: pd.Series):
    peak = cum_series.cummax()
    dd = cum_series / peak - 1
    mdd = dd.min()
    return dd, mdd

str_dd, str_mdd = compute_drawdown(result["cum_return"])
kos_dd, kos_mdd = compute_drawdown(kospi_cum)

# Annual metrics (yearly return and yearly Sharpe)
years = sorted(set(daily_index.year))
annual_rows = []
for y in years:
    mask = daily_index.year == y
    if mask.sum() == 0:
        continue
    s_ret = daily_result.loc[mask, "portfolio_return"].dropna()
    k_ret = kospi_ret.loc[mask].dropna()

    s_annual_return = (1 + s_ret).prod() - 1 if len(s_ret) > 0 else None
    k_annual_return = (1 + k_ret).prod() - 1 if len(k_ret) > 0 else None

    s_sharpe = (s_ret.mean() / s_ret.std() * (252 ** 0.5)) if (len(s_ret) > 1 and s_ret.std() != 0) else None
    k_sharpe = (k_ret.mean() / k_ret.std() * (252 ** 0.5)) if (len(k_ret) > 1 and k_ret.std() != 0) else None

    annual_rows.append({
        "year": y,
        "strategy_annual_return": s_annual_return,
        "kospi_annual_return": k_annual_return,
        "strategy_sharpe": s_sharpe,
        "kospi_sharpe": k_sharpe,
    })

annual_df = pd.DataFrame(annual_rows).set_index("year")

# Overall metrics
overall_metrics = {
    "strategy_mdd": float(str_mdd),
    "kospi_mdd": float(kos_mdd),
    "strategy_final_cum_return": float(result["cum_return"].iloc[-1]),
    "kospi_final_cum_return": float(kospi_cum.iloc[-1])
}

# Save comparison Excel with multiple sheets
excel_path = output_dir / f"benchmark_{args.factor}_{args.group}_{args.allocation}.xlsx"
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    daily_compare.to_excel(writer, sheet_name="daily", index=True)
    annual_df.to_excel(writer, sheet_name="annual_metrics")
    pd.DataFrame([overall_metrics]).to_excel(writer, sheet_name="overall")

print(f"✓ Benchmark Excel saved: {excel_path}")

# Plots: cumulative return, drawdown, annual return & sharpe
print("Generating benchmark plots...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# 1) cumulative returns
ax = axes[0]
ax.plot(result["cum_return"].index, result["cum_return"].values, label="Strategy", linewidth=2)
ax.plot(kospi_cum.index, kospi_cum.values, label="KOSPI", linewidth=2, linestyle="--")
ax.set_title("Cumulative Return: Strategy vs KOSPI")
ax.legend()
ax.grid(True, alpha=0.3)

# 2) drawdowns
ax = axes[1]
ax.plot(str_dd.index, str_dd.values, label="Strategy DD")
ax.plot(kos_dd.index, kos_dd.values, label="KOSPI DD")
ax.set_title("Drawdown (negative = drawdown)")
ax.legend()
ax.grid(True, alpha=0.3)

# 3) annual returns
ax = axes[2]
annual_df[["strategy_annual_return", "kospi_annual_return"]].plot(kind="bar", ax=ax)
ax.set_title("Annual Return by Year")
ax.grid(True, alpha=0.3)

# 4) annual sharpe
ax = axes[3]
annual_df[["strategy_sharpe", "kospi_sharpe"]].plot(kind="bar", ax=ax)
ax.set_title("Annual Sharpe by Year")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = output_dir / f"benchmark_{args.factor}_{args.group}_{args.allocation}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"✓ Benchmark plot saved: {plot_path}")

# Append benchmark metrics to markdown report
try:
    n_days = kospi_ret.dropna().shape[0]
    kospi_cagr = kospi_cum.iloc[-1] ** (252.0 / n_days) - 1 if n_days > 0 else None
    kospi_sharpe = (kospi_ret.mean() / kospi_ret.std() * (252 ** 0.5)) if (n_days > 1 and kospi_ret.std() != 0) else None
    kospi_mdd = float(kos_mdd)

    rpt_path = output_dir / "report.md"
    with open(rpt_path, "a", encoding="utf-8") as f:
        f.write("\n---\n\n## Benchmark Performance (KOSPI)\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| KOSPI CAGR | {kospi_cagr:.4f} |\n" if kospi_cagr is not None else "| KOSPI CAGR | N/A |\n")
        f.write(f"| KOSPI Sharpe | {kospi_sharpe:.4f} |\n" if kospi_sharpe is not None else "| KOSPI Sharpe | N/A |\n")
        f.write(f"| KOSPI MDD | {kospi_mdd:.4f} |\n")
    print(f"✓ Appended benchmark metrics to: {rpt_path}")
except Exception as e:
    print(f"! Failed to append benchmark metrics to report: {e}")

print("\nAll outputs saved to:")
print(output_dir.resolve())