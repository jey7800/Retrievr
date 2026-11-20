# 0006 — Snapshots and benchmark layout

Date: 2026-11-20 · Status: accepted

## Context

Tests and the benchmark must run offline and reproducibly, from captures of real sites.
Captured pages are third-party content; document files are large and not needed to
measure listing recall.

## Decision

- **Snapshot format v1** (`retrievr.snapshots`): a directory with `index.yaml` (one entry
  per GET: URL, status, content type, `Location`, ETag, Last-Modified, SHA-256, size,
  capture time) and the bodies of text responses under `pages/`. Document bodies are never
  stored. Recording is an httpx response hook on the `Fetcher`, so it records robots.txt,
  each redirect hop and every page exactly as the library fetched them, also behind a
  proxy. Replay is an httpx transport; documents are served as small synthetic files
  with valid magic bytes; unknown URLs answer 404 and are reported as misses. A
  `Fetcher` given a custom transport ignores environment proxies.
- **Benchmark** (`bench/`, never part of `pytest`): `sites.yaml` (every site with its
  qualification profile), `ground_truth/<site>.yaml` (pydantic-validated: written scope,
  fiscal-year window, documents, three natural-language requests with their expected
  subset as a filter, `reviewed` flag), `capture.py` (live capture, opt-in) and `run.py`
  (recall, precision in the window, metadata accuracy, snapshot coverage, time, LLM calls,
  cost).
- **What is committed:** snapshots used by `pytest` (`tests/fixtures/`) are committed in
  full. Benchmark snapshots (`bench/snapshots/<site>/`) commit only `index.yaml` (with
  hashes, to detect drift); their `pages/` are git-ignored and recaptured locally with
  `bench/capture.py`. Results (`bench/results/`) are git-ignored.

## Consequences

The offline benchmark needs a local capture first on a fresh clone. Sites whose lists
come from JavaScript (profile `js_xhr`) have a ground truth but no complete snapshot
until browser capture exists.
