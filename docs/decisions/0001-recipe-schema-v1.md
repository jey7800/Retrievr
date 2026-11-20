# 0001 — Recipe schema v1

Date: 2026-11-20 · Status: accepted

## Context

A recipe describes a site (where the document listings are, how to paginate, how to
read and type items), never a user request. It must be data that a pydantic schema can
validate, because later milestones will have an LLM propose recipes.

## Decision

`schema_version: 1`, validated by `retrievr.recipe.Recipe` (`extra="forbid"` everywhere):

- identity: `recipe`, `version`, `issuer`, `site`, `allowed_domains` (exact host match;
  `site` must be listed);
- `sources`: one kind, `url_template`, with `{name}` placeholders matching `params`
  (integer year ranges, `to: current`), optional `pagination` (`next_link` only, with
  `max_pages`), and `items` (`container`, `item`, `link`, `title` CSS selectors, a
  `date` regex + `strptime` format, an `extensions` filter);
- `classify`: ordered regex rules for `type`, `period` (named groups `year` + `q`/`h`/`fy`)
  and `language` (group `lang` or a fixed value), each with `target: url | text | any`;
- `defaults.language`;
- `health` (anchors, `items_per_year`, fingerprint, cadence) and `provenance` are stored
  but not evaluated yet.

Selectors and regexes are compiled at load time, so an invalid recipe never runs.
Pagination only follows the link matched by the selector; page numbers are never guessed
(a real site returns unrelated results for an out-of-range page).

Two deliberate differences from earlier sketches:

- the rule field is `target`, not `on`: YAML 1.1 parses a bare `on` key as `true`;
- no `engine` version range: `schema_version` is the compatibility contract, and a
  recipe cannot change fetch behaviour (rate limit, robots) — only the host
  configuration can.

Document type ids are free `snake_case` strings for now (taxonomy not frozen).

## Consequences

Adding a source kind, a pagination kind or a parameter type is a schema change: bump
`schema_version` and write an ADR.
