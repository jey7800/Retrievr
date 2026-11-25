# 0015 — Reset history on install, count skipped items, name sources

Date: 2026-11-25 · Status: accepted · Extends 0014

## Context

Running a schema v6 recipe on a real host session showed three gaps:

1. A listed item whose identity values (`download.identity`) are not all present was
   dropped without a trace. A portal that leaves one field blank on a whole category made
   that category vanish from the results, with nothing in the warnings.
2. A new recipe version that changes how documents are identified (another `identity`)
   makes every known document look gone: the health check reads it as drift and the site
   ends `broken`. The only way out was editing the internal SQLite state by hand, which is
   not a public interface.
3. The `run` JSON gives no source per document, so a host counting documents per source
   had to rebuild that from URLs. Warnings of the recipe run itself (`max_pages` reached,
   listing stopped early) were also not passed on to `run`'s warnings.

## Decision

- Items skipped because an identity value is missing are counted per source and reported
  as one warning: `<source>: <n> items skipped, identity value missing: <names>` (value
  names only, never values).
- `Retrievr.install(path, reset_history=True)` and `retrievr install <file> --reset-history`
  mark the documents of the site's history as removed before the run that follows. Rows
  are kept (soft removal); downloaded files and the manifest are untouched, so files keep
  deduplicating by content. The `install` JSON says `"history": "reset"`.
- Each document of the `run` JSON has `source` (the recipe source id). Warnings of the
  recipe run join the `run` warnings.

No recipe schema change; the public API gains one keyword argument with a default.

## Consequences

- A host that changes identities installs the new version with `--reset-history` once;
  the next run rebuilds the history and its health is `ok`.
- Resetting hides real drift for that one run: use it only together with a new version
  whose identities differ.
