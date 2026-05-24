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

## CLI Usage

Run a single strategy:

```powershell
python run.py --factor pbr --group Sector --direction low --weighting rank --allocation signal --top-pct 0.2 --max-weight 0.1 --reb-freq M --transaction-cost 0.0025 --data-path ../data --output-dir output --report-name strategy_report
```

Run multiple strategies in one command:

```powershell
python run_multi.py --strategy "PBR_SECTOR|pbr|Sector|low|rank|signal|0.2|0.1|M|0.0025" --strategy "ROE_MARKET|roe|Market|high|score|markowitz|0.15|0.08|M|0.0025" --data-path ../data --output-dir output --report-name multi_strategy_report
```

You can also inspect available options with:

```powershell
python run.py --help
python run_multi.py --help
```

### Generate Excel report from `run.py`

`run.py`는 Markdown/DOCX 보고서와 함께 Excel 파일도 생성할 수 있습니다.

```powershell
python run.py --factor pbr --group Sector --direction low --weighting rank --allocation signal --top-pct 0.2 --max-weight 0.1 --reb-freq M --transaction-cost 0.0025 --data-path ../data --output-dir output --report-name strategy_report --excel-output output/strategy_report.xlsx
```

### Use an Excel template

기존 템플릿을 읽고 새로운 워크북으로 저장하려면 `--excel-template`을 사용하세요.

```powershell
python run.py --factor pbr --group Sector --direction low --weighting rank --allocation signal --top-pct 0.2 --max-weight 0.1 --reb-freq M --transaction-cost 0.0025 --data-path ../data --output-dir output --report-name strategy_report --excel-template PAPERS_리스크팀_인계.xlsx --excel-output output/strategy_report.xlsx
```

### Generate a workbook from a template and add CSV sheets with `run_excel.py`

`run_excel.py`는 템플릿 Excel 파일을 복사하고 CSV 데이터를 별도 시트로 붙여넣는 용도입니다.

```powershell
python run_excel.py --template PAPERS_리스크팀_인계.xlsx --output output/template_report.xlsx --sheet "Performance Summary=output/performance_summary.csv"
```

# Quant-Project