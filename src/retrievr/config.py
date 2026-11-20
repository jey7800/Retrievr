"""Configuration loaded from ``retrievr.toml`` (stdlib ``tomllib``), validated by pydantic."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from retrievr import __version__

DEFAULT_USER_AGENT = f"retrievr/{__version__} (+https://pypi.org/project/retrievr/)"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FetchSettings(_Section):
    user_agent: str = Field(default=DEFAULT_USER_AGENT, min_length=1)
    rate_limit_rps: float = Field(default=1.0, gt=0, le=10)
    timeout_s: float = Field(default=30.0, gt=0)
    max_redirects: int = Field(default=10, ge=0, le=30)
    robots: Literal["respect", "ignore"] = "respect"
    """``ignore`` is an explicit opt-out; it is logged at WARNING level on every run."""
    robots_unreachable: Literal["allow", "deny"] = "allow"
    """Policy when robots.txt cannot be obtained (redirect loop, network error).
    HTTP 4xx means "no robots.txt" (allow) and HTTP 5xx means "disallow", per RFC 9309."""
    max_download_bytes: int = Field(default=500 * 1024 * 1024, gt=0)


class OutputSettings(_Section):
    output_dir: Path = Path("downloads")


StructuredMode = Literal["auto", "json_schema", "tools", "json_object"]
CacheMode = Literal["use", "replay", "off"]


class LLMSettings(_Section):
    """Any OpenAI-compatible chat completions endpoint (configuration only)."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = Field(default="RETRIEVR_LLM_API_KEY", min_length=1)
    """Name of the environment variable holding the key (the key itself is never stored)."""
    model_small: str = "openai/gpt-6-luna"
    model_large: str = "openai/gpt-6-sol"
    structured: StructuredMode = "auto"
    """``auto``: json_schema, then a forced tool call, then json_object + validation."""
    extra_body: dict[str, Any] = Field(
        default_factory=lambda: {"provider": {"require_parameters": True}}
    )
    """Merged into every request body (provider routing, reasoning effort...)."""
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=16000, gt=0)
    timeout_s: float = Field(default=180.0, gt=0)
    max_cost_per_run_usd: float = Field(default=0.10, ge=0)
    prices: dict[str, tuple[float, float]] = {}
    """USD per million tokens (input, output), used when the endpoint returns no cost."""
    cache_dir: Path | None = None
    """Response cache directory (default: ``<state_dir>/llm_cache``)."""
    cache: CacheMode = "use"
    """``use`` reads and writes the response cache, ``replay`` only reads it (a miss is an
    error, used by tests), ``off`` disables it."""


class StateSettings(_Section):
    state_dir: Path = Path(".retrievr")
    """Learned recipes, SQLite state (runs, LLM ledger, decisions) and the LLM cache."""


class Settings(_Section):
    fetch: FetchSettings = FetchSettings()
    output: OutputSettings = OutputSettings()
    llm: LLMSettings = LLMSettings()
    state: StateSettings = StateSettings()


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from a TOML file; defaults when ``path`` is None."""
    if path is None:
        return Settings()
    with Path(path).open("rb") as fh:
        data = tomllib.load(fh)
    return Settings.model_validate(data)
