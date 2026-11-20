# 0005 — Canonical document types

Date: 2026-11-20 · Status: accepted

## Context

The benchmark compares the types a recipe (hand-written now, learned later) gives to
documents with a hand-made ground truth, and later the request filters select documents
by type. Free-form type ids per recipe make those comparisons impossible.

## Decision

`retrievr.taxonomy.DOCUMENT_TYPES` lists the canonical ids: `urd`, `annual_report`,
`half_year_report`, `quarterly_report`, `pillar3`, `results_presentation`,
`fixed_income_presentation`, `press_release`, `financial_statements`,
`financial_data_supplement`, `quarterly_series`, `quarterly_series_restated`,
`analyst_call_transcript`, `fixed_income_call_transcript`, `other`.

Ground-truth files must use them (validated). Periods are written `2025`, `2025-Q3`,
`2025-H1` or `2025-FY`; the fiscal year is the year the document covers, even when it is
published the following year.

Labelling conventions used in the ground truth:

- an earnings release filed as a 6-K/8-K is a `press_release`; a 10-Q is a
  `quarterly_report`, a 10-K or 20-F an `annual_report`;
- an ESEF/XBRL package of a registration document or annual report has the type of that
  document;
- prepared remarks, management speeches and Q&A transcripts are `analyst_call_transcript`;
- Pillar 3 annexes (capital composition, own funds instruments) are `pillar3`;
- restated or historical data packs are `quarterly_series_restated`.

Recipes are not forced to use the taxonomy yet (schema v2 accepts any `snake_case` id).

## Consequences

Adding a type is a taxonomy change: update this ADR and the ground truths that need it.
