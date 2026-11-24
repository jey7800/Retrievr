# 0013 — Recipe schema v5: listings held by the page; fifth round of cold tests

Date: 2026-11-24 · Status: accepted · Extends 0008, 0011

## Context

A fifth round of live cold tests ran on twelve sites the library had never seen (DNB,
Raiffeisen Bank International, Scotiabank, ANZ, Sanofi, Engie, Novo Nordisk, Volvo,
Carrefour, Atlas Copco, Renault, Philips), same request and years as before. Bankinter,
Sabadell, Iberdrola, Air Liquide, Unilever and Julius Baer answer 403 to an identifiable
client and were left out.

| Site | Outcome |
|---|---|
| DNB (1015 listed, 72 downloaded), Engie (60, 36), Philips (44, 33), Atlas Copco (103, 28), ANZ (297, 27), Novo Nordisk (109, 24), RBI (205, 22), Carrefour (27, 20), Sanofi (41, 16) | learned first time, warm run `ok` with 0 LLM calls |
| Scotiabank | the page holds every year as JSON in a data attribute; its year selector filters it without any request: supported below (339 listed, 35 downloaded) |
| Renault | `rejected`: the files host answers 403 (the gate of round 4) |
| Volvo | the start page lists the current period only, its archive is elsewhere: the zero-selection warning says so |

## Decision

Recipe schema **v5** (v1–v4 recipes stay valid and unchanged; learned recipes stay v4
unless they use a v5 feature):

- **`response.embedded`** on a request source (`format: json`): the response is an HTML
  page and the JSON is the value of `attribute` of the elements `selector` matches, or
  their text (`<script type="application/json">`); the first element holding JSON is read.
  The source is a GET of the page itself.
- **`*` in `items.path`**: every key (or list element) at that level; lists spread over
  keys (`{"2025": {"Q1": [...]}}`) are one listing, `*.*`. The keys a `*` goes through
  join the item's context, so rules read the year and quarter.

Learning finds such JSON without a browser: JSON scripts, and `data-*`, `value` or
`content` attributes holding at least 200 characters of JSON, analysed like captured
calls. A block is offered to the LLM (as a "J" pseudo-call next to the browser's "X"
calls) only when it lists at least three documents the page does not link. Lists of the
same depth that differ by at most two keys are offered as one wildcard list.

## Consequences

The trend of first-time success on reachable sites: 3/8, 7/8, 9/12, 8/12, 9/12. This
round needed one new feature (a listing pattern no recipe could describe) and no fix of
existing behaviour. Known limits: a start page that shows the current period only and
links its archive nowhere (Volvo); items carrying a second file (an Excel next to the PDF)
in another field give only the first URL present.
