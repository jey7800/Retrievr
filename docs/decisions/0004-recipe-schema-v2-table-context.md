# 0004 — Recipe schema v2: table context and field targets

Date: 2026-11-20 · Status: accepted · Extends 0001

## Context

Some investor-relations pages list documents in tables: one table per year, one
column per document category, one row per quarter. On such pages the link text is
often just "PDF", the quarter is not always in the file name, and the quarter label
appears only on the first row of a quarter (the next row is blank). Schema v1 could
only read the item text and the URL, which is not enough to type and date these links.

## Decision

`schema_version: 2` adds, as optional features:

- `items.table_context: true`: for links inside a table cell, the candidate gets a
  `row_header` (first cell of the row, carried down over blank cells, rowspan-aware)
  and a `column_header` (header cell above the link's column, colspan-aware). Nested
  tables are handled separately.
- Rules may target a named field instead of `url`/`text`/`any`: `title`, `link_text`,
  `link_title` (the link's `title` attribute), `context`, `heading`, `row_header`,
  `column_header`, or a list of 1-4 fields searched as one string joined with `" | "`.
  Example: `{pattern: '^Q(?P<q>[1-4]) \| (?P<year>20\d{2})$', target: [row_header, heading]}`.

v1 recipes stay valid unchanged. A recipe that declares `schema_version: 1` and uses
a v2 feature is rejected. `.xlsb` files are accepted and checked like other Office
Open XML packages.

## Consequences

Field names are part of the schema: adding or renaming one needs a new version.
