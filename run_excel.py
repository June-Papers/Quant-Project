import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from quant.report.excel_report import read_excel_template, write_excel_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an Excel template and output a new Excel report workbook"
    )
    parser.add_argument("--template", type=str, required=True, help="Path to the Excel template file")
    parser.add_argument("--output", type=str, default="output/excel_report.xlsx", help="Output Excel file path")
    parser.add_argument(
        "--sheet",
        action="append",
        default=[],
        help="Add a CSV file as a sheet in NAME=PATH format, e.g. Summary=performance_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = Path(args.template)
    output_path = Path(args.output)
    template_data = read_excel_template(template_path)

    tables: Dict[str, pd.DataFrame] = {}
    for sheet_def in args.sheet:
        if "=" in sheet_def:
            sheet_name, csv_path = sheet_def.split("=", 1)
            tables[sheet_name.strip()] = pd.read_csv(Path(csv_path.strip()))
        else:
            raise ValueError("Sheet definitions must be in NAME=PATH format.")

    write_excel_report(output_path, tables=tables, template_data=template_data)
    print(f"Excel report generated: {output_path}")


if __name__ == "__main__":
    main()
