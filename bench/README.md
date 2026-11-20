# Benchmark

Opt-in evaluation of retrievr on real investor-relations sites. Never part of `pytest`.

| File | Content |
| --- | --- |
| `sites.yaml` | Every site with its start page and qualification profile (static, JS/XHR, blocked…) |
| `ground_truth/<site>.yaml` | Hand-made ground truth: scope, fiscal years, documents, 3 requests |
| `snapshots/<site>/` | Captured pages (`index.yaml` is committed, `pages/` is not) |
| `capture.py` | Live capture into a snapshot (hits the network) |
| `run.py` | Runner: recall, precision, metadata accuracy, coverage, time, LLM calls, cost |

```bash
# capture (or refresh) a snapshot
uv run python -m bench.capture --site airbus --url "https://www.airbus.com/en/investors/annual-reports"
# offline run on all sites with a ground truth
uv run python -m bench.run
# live run (monthly drift measurement)
uv run python -m bench.run --mode live
```

Metrics, per site:

- **Recall**: ground-truth documents returned / ground-truth documents.
- **Precision**: returned documents that are in the ground truth / returned documents in
  the fiscal-year window (documents without a fiscal year count as in the window).
- **Type, period, language**: share of matched documents with the right value.
- **Coverage**: share of ground-truth documents linked from the snapshot pages. 100 %
  means a recipe or a learner can be evaluated offline on that site.

A ground truth is drafted from what a visitor sees on the page and is not trusted until a
human sets `reviewed: true`. Document types follow `retrievr.taxonomy`
(`docs/decisions/0005-document-taxonomy.md`).
