# 0007 — LLM client and cold learning

Date: 2026-11-20 · Status: accepted

## Context

Milestone 3 lets retrievr learn a recipe for an unknown site from a start URL, with an
LLM used only on cold paths. The LLM must never produce code, page content must be
treated as untrusted, every decision must be logged and replayable offline, and the
cost of a run must be capped.

## Decision

- **Client** (`retrievr.llm`): plain httpx `POST {base_url}/chat/completions`, not the
  `openai` SDK. The SDK now ships its own HTTP stack (`httpx2`), which would mean a second
  proxy and trust-store setup and calls that `respx` cannot mock. Same OS trust store and
  proxy variables as the fetcher; no new dependency.
- **Structured outputs**: strict `json_schema`, then a forced `tools` call, then
  `json_object` with the schema in the prompt (`structured = "auto"`); the strategy that
  works is remembered per model. The schema sent is strict-mode friendly (every property
  required, no constraint keywords); pydantic validates the reply, and an invalid reply is
  retried once with the validation error.
- **Cache and ledger**: validated outputs are cached on disk by SHA-256 of model + JSON
  schema + messages. `cache = "replay"` makes a miss an error (tests). Every call, cached
  or not, is written to the SQLite ledger (`llm_calls`: messages, response, tokens, cost,
  latency, strategy). Cost comes from the endpoint (`usage.cost`) or a configured price
  table. `max_cost_per_run_usd` (default 0.10) is checked before every network call.
- **Learning** (`retrievr.learn`), three steps, each a JSON object the engine checks:
  1. *Structure*: the LLM sees a page digest (document-link groups by DOM path with
     digits folded, navigation links, pruned HTML) and either asks to open listed links
     (ids only, same site, at most 5 per round and 2 rounds) or proposes sources (URL
     with an optional `{year}` placeholder, container, item, pagination). The engine
     builds a recipe, replays it and compares with the groups the LLM selected; missing
     or extra documents go back as a diagnostic (2 repairs at most).
  2. *Rules*: the LLM writes type, period and language rules and labels a sample of
     documents (at most 150). The engine measures how often the rules reproduce the
     labels (target 90 %) and asks for one repair otherwise. Labels become stored
     decisions for documents the rules leave undecided.
  3. *Request*: the request is compiled once into a `Filter` (types, excluded types,
     fiscal-year range, languages), cached in SQLite by hash of the request and year.
- **Types are closed**: LLM schemas use `taxonomy.DocumentType`, so the LLM cannot invent
  a type. The recipe schema itself is unchanged (still v2).
- **LLM fallback** (`classify.classify_with_llm`): documents that rules and stored
  decisions leave without type or period, within the requested years, are classified in
  batches of 50; decisions are stored, so a warm run on an unchanged site makes no call.
- **Validation before activation** (`retrievr.validate`): the run succeeds, finds at least
  one document, stays in the allowed domains, and two consecutive runs agree. The gates
  that need history (anchors, 95 % of known documents, 1.5x volume) are implemented and
  apply from repair on (milestone 5). A rejected recipe is not saved or activated and a
  `recipe_rejected` event is emitted.
- **Storage** (`retrievr.store`): `<state_dir>/recipes/<recipe>@<version>.yaml`, written
  once, plus SQLite (`active`, `runs`, `llm_calls`, `decisions`, `filters`,
  `site_requests`).
- **Interpreter change**: a 404 or 410 on a listing page built from a parameter (a year
  not published yet) is a warning, not a run error.

## Consequences

Learning a site costs a few LLM calls (about 0.5 US cent with a small model) and is
replayable offline from the cache. Opaque links, JavaScript listings and POST endpoints
are still out of reach (milestone 4); health checks and repair come in milestone 5.
