"""Canonical document types (ADR 0005).

Recipes may still use other ``snake_case`` ids; the benchmark ground truth, and later
the compiled request filters, use these.
"""

from __future__ import annotations

import re
from typing import Literal, get_args

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


DocumentType = Literal[
    "urd",
    "annual_report",
    "half_year_report",
    "quarterly_report",
    "pillar3",
    "results_presentation",
    "fixed_income_presentation",
    "press_release",
    "financial_statements",
    "financial_data_supplement",
    "quarterly_series",
    "quarterly_series_restated",
    "analyst_call_transcript",
    "fixed_income_call_transcript",
    "other",
]
"""The same ids as a type, for schemas the LLM must fill (it cannot invent a type)."""

if set(get_args(DocumentType)) != set(DOCUMENT_TYPES):  # pragma: no cover
    raise RuntimeError("DocumentType and DOCUMENT_TYPES differ")

PERIOD = re.compile(r"^(?P<year>\d{4})(-(Q[1-4]|H[12]|FY))?$")
"""Canonical period: ``2025``, ``2025-Q3``, ``2025-H1`` or ``2025-FY``."""

CONVENTIONS = """Labelling conventions:
- An earnings release filed as a 6-K or 8-K is a press_release; a 10-Q is a
  quarterly_report; a 10-K or 20-F is an annual_report.
- An ESEF/XBRL package of a registration document or report has the type of that document.
- Prepared remarks, management speeches and Q&A transcripts are analyst_call_transcript.
- Pillar 3 annexes (capital composition, own funds instruments) are pillar3.
- Restated or historical data packs are quarterly_series_restated.
- An amendment to a universal registration document is a urd.
Periods: the period the document covers, written 2025-Q3 (quarter), 2025-H1 (half
year), 2025-FY (full year or annual document) or 2025 (year only, when nothing finer is
known). The fiscal year is the year covered, even when the document is published the
following year. A fourth-quarter document labelled "Q4"/"4Q" is 2025-Q4; one labelled
only "full year"/"FY"/"annual" is 2025-FY. Annual results documents (annual results
presentations, annual reports, registration documents, annual financial statements) are
2025-FY even when only the year is shown; the year alone is for documents that cover no
reporting period (a policy, a tax strategy)."""


def is_known_type(value: str) -> bool:
    return value in DOCUMENT_TYPES


def taxonomy_text() -> str:
    """The taxonomy as prompt text."""
    lines = [f"- {key}: {label}" for key, label in DOCUMENT_TYPES.items()]
    return "Document types:\n" + "\n".join(lines) + "\n" + CONVENTIONS


def canonical_period(value: str | None) -> str | None:
    """``value`` if it is a canonical period, else None."""
    if value is None:
        return None
    text = value.strip().upper()
    return text if PERIOD.match(text) else None
