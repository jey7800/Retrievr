"""Canonical document types (ADR 0005).

Recipes may still use other ``snake_case`` ids; the benchmark ground truth, and later
the compiled request filters, use these.
"""

from __future__ import annotations

DOCUMENT_TYPES: dict[str, str] = {
    "urd": "Universal registration document (or equivalent registration document)",
    "annual_report": "Annual report, annual financial report, 10-K style report",
    "half_year_report": "Half-year (interim) financial report",
    "quarterly_report": "Quarterly report, earnings report, quarterly full report",
    "pillar3": "Pillar 3 risk disclosure",
    "results_presentation": "Results presentation slides (equity analysts)",
    "fixed_income_presentation": "Fixed income / debt investor presentation",
    "press_release": "Results press release, media release",
    "financial_statements": "Financial statements published on their own",
    "financial_data_supplement": "Financial data supplement (PDF or spreadsheet)",
    "quarterly_series": "Quarterly data series (usually a spreadsheet)",
    "quarterly_series_restated": "Restated quarterly data series",
    "analyst_call_transcript": "Transcript or prepared remarks of the analyst call",
    "fixed_income_call_transcript": "Transcript of the fixed income call",
    "other": "Any other document in scope",
}


def is_known_type(value: str) -> bool:
    return value in DOCUMENT_TYPES
