
# Quant Backtest Report

## Strategy Information

| Item | Value |
|---|---|
| Factor | roe |
| Group Neutralization | Sector |
| Direction | high |
| Weighting | rank |
| Allocation Method | markowitz |
| Max Weight | 0.07 |
| Rebalance Frequency | M |
| Transaction Cost | 0.0025 |

---

## Performance Summary

| Metric | Value |
|---|---|
| CAGR | 0.0181 |
| Sharpe | 0.1924 |
| MDD | -0.6251 |
| Turnover(mean) | 0.0506 |


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

---

## Benchmark Performance (KOSPI)

| Metric | Value |
|---|---|
| KOSPI CAGR | 0.0716 |
| KOSPI Sharpe | 0.4422 |
| KOSPI MDD | -0.5454 |
