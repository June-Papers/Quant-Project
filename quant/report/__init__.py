from .docx_report import ReportAssembler
from .charts import generate_report_charts
from .tables import (
    build_strategy_overview,
    build_performance_table,
    build_phase_performance_table,
    build_factor_excess_table,
)
from .utils import (
    build_annual_return_table,
    build_comparison_performance_table,
    build_phase_comparison_table,
    build_report_overview,
    compute_monthly_returns,
    compute_phase_labels,
    compute_phase_summary,
)

__all__ = [
    "ReportAssembler",
    "generate_report_charts",
    "build_strategy_overview",
    "build_performance_table",
    "build_phase_performance_table",
    "build_factor_excess_table",
    "build_annual_return_table",
    "build_comparison_performance_table",
    "build_phase_comparison_table",
    "build_report_overview",
    "compute_monthly_returns",
    "compute_phase_labels",
    "compute_phase_summary",
]
