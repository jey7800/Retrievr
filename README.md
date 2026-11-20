# retrievr

Find and download documents (PDF, XLSX…) from websites, such as the investor-relations
pages of listed companies, with an LLM used **only when needed**.

> Status: early development (deterministic core, benchmark, LLM learning).

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

The benchmark (`bench/`, opt-in) measures recall on real sites against hand-made ground
truth; see `bench/README.md`.

## License

Apache-2.0.
