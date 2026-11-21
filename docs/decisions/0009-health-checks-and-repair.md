# 0009 — Health checks and the repair ladder

Date: 2026-11-21 · Status: accepted · Extends 0007, 0008

## Context

Sites change: classes and ids are renamed, URL parameters change, pagination becomes a
"load more" link, a listing moves into a JSON API. A recipe that silently lists fewer
documents, or misreads them, is worse than one that fails. Repair must stay automatic
(no human approval), cheap (deterministic first), and safe: nothing is activated
without passing validation against what the site was known to publish.

## Decision

- **History** (SQLite, internal): every document a healthy run lists, with its listing
  parameters (year, option) and its type, period and language; health checks; repair
  attempts; the site's status (`ok` or `degraded`) and start URL.
- **Health check** (`retrievr.health.assess`, no LLM), compared with the history of the
  years the run covers:
  - `blocked`: the site refused us;
  - `broken`: no document, an anchor missing, more than 5 % of known documents missing,
    more than 10 % of known documents with another type or period, a listing URL that
    answers 404/410, a response the recipe cannot read;
  - `suspect`: at least one known document missing or changed (below the broken
    thresholds), 1.5x more documents than known, documents per year outside
    `health.items_per_year`, the site unreachable;
  - `stale`: the latest publication is older than `health.cadence.max_gap_days`;
  - `ok` otherwise. Missing documents are checked online (a streamed GET of at most 5):
    documents gone from the site (404/410) are marked removed and are not a breakage.
  This makes a false `ok` impossible when a known document disappears from a listing
  while it is still online.
- **Repair** (`retrievr.repair`), only on `broken`:
  1. *Relocation*, deterministic: the known documents are looked for in the pages as they
     are now and the parts that stopped matching are rebuilt around them: a renamed year
     parameter in the listing URL (from the links of the page), the `<select>` whose
     option values are those the documents were listed under, container / item / title /
     link selectors (an id first, then the most distinctive class), the "next page" link
     (`rel=next`, a link to the next page number, a "next" / "load more" text). Variants
     are replayed and kept when they find more known documents; a part that matches
     nothing at all is replaced even before the recipe finds anything again.
  2. *Small model*: the cold-path learner runs again with a repair hint (the previous
     sources, the failed checks, the documents to find again, which are added to the
     documents its structure loop must find).
  3. *Large model*: the same with the large tier, skipped when it is the small model.
  4. *Degraded*: the old recipe stays active, the site is marked `degraded`, the
     `site_degraded` event is emitted and the run downloads nothing.
- A repaired recipe is the next version of the old one (immutable files, provenance
  `repaired`), keeps its classification rules and health expectations, and must pass
  the validation gates with the history: all anchors still online, at least 95 % of the
  known documents still online, at most 1.5x their number (known documents of other
  years do not count), allowed domains, two identical runs, and **at least 90 % of the
  known documents keeping their type and period** (new gate). Known documents keep their
  previous metadata as stored decisions; rules still win where they decide.
- **Budget**: at most 2 LLM repair attempts (rungs 2 and 3) per site per day; relocation
  is free. The cost cap of the run (0.10 USD by default) applies to repairs too.
- **API**: `run(..., repair=True)` checks health after every replay and repairs a broken
  recipe before returning documents; `check(site)` is the health check alone (no LLM,
  no download); `repair(site)` runs the ladder on demand; `install(path)` activates a
  hand-written recipe. `RunOutcome` gains `health` and `repair`, its status `degraded`.
  New events: `health_alert`, `recipe_repaired`, `site_degraded` (and `recipe_rejected`
  for each candidate that failed validation). The recipe schema is unchanged.

## Consequences

Mutation tests (`tests/mutations/`) break the snapshots of four sites in seven ways.
Results with gpt-6-luna (replayed offline from the response cache):

- 14 cases; zero false `ok`: the 9 mutations that changed the old recipe's output are all
  detected (`broken`), and the 5 that did not (extra wrappers, reordered table columns)
  stay `ok`;
- 7 of 7 repairable breakages repaired: renamed classes and ids (Société Générale,
  Crédit Agricole), renamed URL parameter and "load more" (Société Générale) by
  relocation alone, the listing moved into a JSON API (Société Générale, Deutsche Bank,
  UBS) by the small model;
- the two unrepairable ones (listing emptied, files still online) end `degraded` with
  the old recipe active: no invalid recipe was activated.

Limits: relocation rebuilds selectors from what the pages show now, so an alternative of
an old selector that no current page uses is dropped (health will flag documents missed
later). When a repaired recipe's rules no longer type new documents (a JSON listing whose
fields the old rules do not read), those documents go to the LLM classification fallback
until rules are learned again.
