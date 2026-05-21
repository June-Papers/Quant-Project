# Quant-Project

Lightweight WorldQuant-style research framework.

Quick start

1. Put your parquet datasets under `../data/` relative to the project root:

   - `close.parquet`, `cap.parquet`, `sector.parquet`, `BS.parquet`, `PL.parquet`

2. Example usage in a notebook or script:

```python
from quant import DataLoader, Backtester, pbr

dl = DataLoader(data_path="../data")
close = dl.load_close()
cap = dl.load_cap()
sector = dl.load_sector()

# build a toy factor (replace with real factor workflow)
# signal should be a DataFrame with 'date' column and stock columns

# run backtest
bt = Backtester(close=close, cap=cap, sector=sector)
result = bt.backtest(signal=close, group='Market')
print(result['cum_return'].tail())
```

See the `quant` package for modular functions: `data.py`, `factor.py`, `backtester.py`, `analytics.py`.
# Quant-Project