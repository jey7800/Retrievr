# retrievr

Find and download documents (PDF, XLSX…) from websites, such as the investor-relations
pages of listed companies, with an LLM used **only when needed**.

> Status: early development (milestone 1, deterministic core). No LLM support yet.

## How it works

A **recipe** (declarative YAML, never code) describes a site: where its document
listings are, how to paginate and how to type each document. The engine replays it with
plain HTTP requests and rules, then downloads the files with integrity checks and
deduplication, into a file tree with a `manifest.json`.

```python
from retrievr.config import load_settings
from retrievr.download import Downloader
from retrievr.fetch import Fetcher
from retrievr.recipe import load_recipe, run_recipe

settings = load_settings("retrievr.toml")          # or load_settings() for defaults
recipe = load_recipe("recipes/societegenerale-ir@1.yaml")
with Fetcher(settings.fetch) as fetcher:
    result = run_recipe(recipe, fetcher, years=[2024, 2025])
    outcomes = Downloader(fetcher, settings.output.output_dir).download_all(
        recipe, result.documents
    )
```

These module-level functions are internal for now; the public `Retrievr` API comes later.

Output: `<output_dir>/<issuer>/<year>/<type>_<period>_<lang>.<ext>` plus
`<output_dir>/manifest.json`.

## Respectful crawling

robots.txt is respected by default (see `docs/decisions/0002-robots-policy.md`), requests
carry an identifiable User-Agent and are rate-limited (1 request per second per host by
default). Blocks (HTTP 403/429/503, challenge or CAPTCHA pages) stop the run with a
`blocked` status; there is no CAPTCHA solving and no evasion.

## Network

Proxy settings come from the standard environment variables (`HTTPS_PROXY`,
`NO_PROXY`…). TLS uses the operating system trust store (`truststore`).

## Development

```bash
uv sync
uv run pytest        # offline: HTML snapshots + respx, never the network
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## License

Apache-2.0.
