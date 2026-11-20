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


class Settings(_Section):
    fetch: FetchSettings = FetchSettings()
    output: OutputSettings = OutputSettings()
    llm: dict[str, Any] | None = None
    """Accepted but not interpreted yet (LLM support comes in a later milestone)."""


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from a TOML file; defaults when ``path`` is None."""
    if path is None:
        return Settings()
    with Path(path).open("rb") as fh:
        data = tomllib.load(fh)
    return Settings.model_validate(data)
