# retrievr

Find and download documents (PDF, XLSX…) from websites, such as the investor-relations
pages of listed companies, with an LLM used **only when needed**.

> Status: early development (deterministic core, benchmark, LLM learning, health checks
> and repair).

## How it works

A **recipe** (declarative YAML, never code) describes a site: where its document
listings are, how to paginate and how to type each document. The engine replays it with
plain HTTP requests and rules, then downloads the files with integrity checks and
deduplication, into a file tree with a `manifest.json`.

On a new site, `learn` asks the LLM to understand the pages once and stores the recipe
it proposes, after the engine has replayed and validated it. Later runs replay the
recipe with no LLM call.

```python
from retrievr import Retrievr

r = Retrievr("retrievr.toml", on_event=print)  # or Retrievr() for defaults
result = r.learn("https://example-issuer.com/investors", "annual reports since 2024")
print(result.status, len(result.documents))
outcome = r.run("example-issuer")  # replays the active recipe, downloads new files
```

## Command line

```bash
retrievr learn https://example-issuer.com/investors "annual and half-year reports" --years 2024-2025
retrievr run example-issuer             # replay, check health, repair if broken, download
retrievr check example-issuer           # health check only: no LLM, no download
retrievr repair example-issuer          # run the repair ladder on demand
retrievr install recipes/example-issuer-ir@1.yaml   # activate a hand-written recipe
```

`--years 2024-2025` (or `2024,2025`) keeps only the documents of those fiscal years, for
the documents returned and downloaded; the recipe itself covers every year the site lists.
Without it, `run` downloads every document the request selects, all years included.
A file deleted from the output directory is downloaded again on the next run.

Settings come from `--config` (else `./retrievr.toml`); `--state-dir` and `--output-dir`
override them. `--json` prints one JSON object for scripts, `-v` prints events and logs
on stderr. Exit codes: 0 success, 1 failure (rejected, degraded, broken), 2 bad usage,
3 blocked by the site, 4 LLM budget exceeded. `python -m retrievr` works too.

## Health checks and repair

Every `run` compares the replay with what earlier runs established (the documents listed,
their type and period, the recipe's anchors and expected volume) and gives a verdict:
`ok`, `stale`, `suspect`, `broken` or `blocked`. No LLM is involved.

When the recipe is `broken`, `run` repairs it before returning documents: it first
rebuilds the selectors, URL or pagination that stopped matching around the documents it
knew (no LLM), then asks the LLM (at most 2 attempts per site per day). A repaired recipe
is activated only if it finds the known documents again with the same metadata;
otherwise the old recipe stays active, the site is marked `degraded` and the host gets a
`site_degraded` event.

```python
report = r.check("example-issuer")  # health check only: no LLM, no download
print(report.verdict, report.summary())
r.install("recipes/example-issuer-ir@1.yaml")  # activate a hand-written recipe
```

## LLM configuration

Any OpenAI-compatible endpoint, by configuration only. The API key is read from the
environment variable named by `api_key_env` (put it in a git-ignored `.env` and run with
`uv run --env-file .env ...`).

```toml
[llm]
base_url    = "https://openrouter.ai/api/v1"
api_key_env = "RETRIEVR_LLM_API_KEY"
model_small = "openai/gpt-6-luna"
structured  = "auto"   # json_schema -> forced tool call -> json_object
max_cost_per_run_usd = 0.10

[state]
state_dir = ".retrievr"  # learned recipes, SQLite state, LLM response cache
```

Every LLM call is logged (prompt, response, tokens, cost) in the SQLite state, and
responses are cached so that tests replay them offline.

Output: `<output_dir>/<issuer>/<year>/<type>_<period>_<lang>.<ext>` plus
`<output_dir>/manifest.json`.

## Pages loaded by JavaScript

When a page fills its listing with JavaScript (year selectors, "load more", a search
API), `learn` opens it once in a browser to see which requests it makes, then stores a
recipe that replays those requests with plain HTTP. The browser is an optional extra and
is never needed to run a recipe:

```bash
pip install "retrievr[browser]"   # Playwright; drives the installed Edge by default
```

The listing may come from an API of another host (an investor-relations content
provider): such read-only (GET) calls are followed when their responses list documents of
the site itself. APIs that need a session, a cookie or an access token are not supported.

```toml
[browser]
channel = "msedge"   # or "chrome", or "" for Playwright's own Chromium
```

The browser is opened when the static page lists few documents, few of the requested
years (it shows the current period only), or has a year selector; also once for an archive
page the LLM opens. A listing API that gives bare file names (`Report_2025.pdf`) is read
with the download URL the page builds from them, taken from the rendered page's links.

## Detail pages

Some sites list one page per event or period ("H1 2025 results", "View details") and put
the documents on those pages. A recipe (schema v4) then names the listing page and the
CSS selector of the links to the detail pages; the engine opens each of them (same site
only) and reads its documents, with the page heading as their context.

## Respectful crawling

robots.txt is respected by default (see `docs/decisions/0002-robots-policy.md`), requests
carry an identifiable User-Agent and are rate-limited (1 request per second per host by
default). Blocks (HTTP 403/429/503, challenge or CAPTCHA pages) stop the run with a
`blocked` status; there is no CAPTCHA solving and no evasion. A listing call that
robots.txt disallows is not made either: `learn` then says so in its notes.

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

The benchmark (`bench/`, opt-in) measures recall on real sites against hand-made ground
truth; see `bench/README.md`.

## License

Apache-2.0.
