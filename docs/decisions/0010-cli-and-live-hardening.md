# 0010 — Command line, and fixes from live tests on new sites

Date: 2026-11-21 · Status: accepted · Extends 0007, 0008, 0009

## Context

The library had only a Python API and had been validated on snapshots of the benchmark
sites. Before a release it was run end to end, live, on investor-relations sites it had
never seen (banks and large companies of several countries), which showed gaps.

## Decision

- **CLI** (`retrievr learn | run | check | repair | install`, also `python -m retrievr`)
  with the standard library's `argparse` (no new dependency; CLAUDE.md's typer target
  can replace it later without changing the commands). `--json` prints one object,
  `-v` prints events and logs on stderr, output is UTF-8 whatever the console code page.
  Exit codes: 0 success, 1 failure, 2 usage, 3 blocked, 4 budget exceeded.
- **Opaque document links** (`/assets/pdf/<name>`, `/files?p_Doc_Ref=<name>`,
  `?disposition=inline`): the learner now sees them (`candidates.OPAQUE_DOCUMENT`) and
  the recipe lists them with `items.document_urls` patterns derived by the engine from
  the groups the LLM selected (the folder of a path, or the URL and query key of a
  download service).
- **URL query fields of a listing API**: a call's query parameters are fields named
  `?name` (e.g. `?page`); the engine templates them in the request URL like body fields.
  A field named without (or with) the `?` is matched leniently. A year inside a value
  keeps its format (`2026-05-01` -> `{year}-05-01`).
- **Year selectors rendered by JavaScript**: when the options selector the LLM names
  matches nothing in the page as the engine fetches it, and the captured values hold
  years, the field varies by year instead.
- **`has_more`** is kept only when the captured response holds a value at that path
  (a missing path would stop pagination after the first page).
- **Field names** given by the LLM are matched leniently: a short name that is the last
  part of exactly one JSON path (`page` for `queries.0.params.page`) is that field.
- **Duplicate request sources** (same call, items and varying fields, another URL path
  in the items) are merged into one source with both paths.
- **Fiscal year fallback**: when the rules give no period, a listing parameter gives the
  year, else a section heading naming exactly one year (a `<h2>2024</h2>` table), else
  the publication date.
- **Periods**: quarters written as words ("fourth quarter", "troisième trimestre") are
  read; the rules step reports rules that give a period not ended yet (impossible for a
  published document, usually a year taken from an upload folder of the URL); the rules
  prompt says that figures "as at" a date report on the period ending that day.
- **Scope**: the structure prompt leaves out separate sections for debt investors
  (credit updates, ratings, debt programmes and prospectuses).
- **Listing APIs of another host** (an IR content provider called by the page): browser
  discovery lets the page's other-site XHR/fetch **GET** calls through (robots.txt of that
  host checked) and the learner accepts them only when every document they list is on
  the site's own domain; that API host is then an allowed domain of the recipe.
  Other-site POSTs (beacons, telemetry) stay blocked.
- **Request filters**: the filter prompt says that "results", "earnings" or "results
  publications" name periods, not a kind of document; the compiled-filter cache key
  includes a hash of the prompt, so an improved prompt recompiles stored requests. A
  request that selects none of the listed documents returns a warning saying which types
  and years the site lists.

## Consequences

Live results with gpt-6-luna, years 2024–2025 (about 0.06 USD of LLM calls in all):

| Site | Listing | Outcome |
|---|---|---|
| Danone | static page, 128 link groups | learned, 159 documents 2014–2026, all typed; warm run and check `ok`, no LLM |
| BBVA | static page, 530 links, year selector | learned, 530 documents; recent years fully typed, older ones partly untyped |
| Nordea | JSON API (GET, `?page`, year inside a filter value) | learned, 77 documents, periods right (a Q4 report published in January keeps its fiscal year) |
| Schneider Electric | opaque links in a table | learned, 13 documents (the page holds the current year only; health reports the missing past year as `suspect`) |
| Commerzbank | listing API of a content provider that needs a token (401) | `rejected`, nothing activated |
| Nestlé | HTTP 403 | `blocked` (exit code 3), no LLM call |

The LLM response cache of the tests was recorded again since prompts changed. Not
supported, by design: APIs that require an authorization token, pages that only render
documents in the browser without a listing call; such sites end `rejected`, no recipe
is activated.

## Addendum (2026-11-23): real-usage review

A review of what a user does day to day found these and fixed them (tests in
`tests/integration/test_real_usage.py`):

- `--years` limits the documents returned and downloaded, not only the listing pages
  replayed: a recipe reading one page that lists every year downloaded them all.
- A request that cannot be compiled (no LLM key, endpoint down) ends the run in
  `error` with nothing downloaded; it used to fall back to every document. A failure
  to classify new documents only leaves them untyped, with a warning.
- A file deleted from the output directory is downloaded again (its manifest entry is
  dropped); the temporary download folder is removed when downloads end.
- The active recipe is kept by file name inside the state directory, so the state can
  be used from any working directory or moved.
- Installing an edited recipe under an installed version is refused (stored versions
  are immutable): raise its `version`.
- `run` ends `budget_exceeded` (exit code 4) when the LLM cap is reached.
- A network error on a GET (reset, timeout, proxy) is retried once after 2 s; a POST is
  never sent twice.
- Messages: an unknown site names the active sites; a start URL without `http(s)://`
  is refused; a missing LLM key says how to pass `.env`. The CLI prints one progress
  line per document fetched (`document_downloaded` event), where the files are, and
  why a document was not downloaded.
- **Downloadable gate** (validation, learning only): up to 3 of the most recent listed
  documents are fetched (streamed, body not read). When they answer 404/410 and the
  listing items carry several URL paths (a storage path and a public link), the other
  paths are tried first and the variant whose documents are reachable is kept; a
  recipe whose sampled documents all answer 404/410 is rejected instead of activated.
  A run or learn where no document could be downloaded returns a warning.
- **Markup documents**: `.xhtml` files (ESEF annual reports in inline XBRL) are
  accepted when served as HTML/XHTML and start with markup; other extensions still
  reject an HTML page.
- The CLI accepts its global options (`-v`, `--json`, `--state-dir`, `--output-dir`,
  `--config`) before or after the command.
