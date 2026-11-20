"""Benchmark data files: site list and hand-written ground truth (pydantic-validated)."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retrievr.taxonomy import DOCUMENT_TYPES
from retrievr.urls import normalize_url

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
PERIOD = re.compile(r"^\d{4}(-(Q[1-4]|H[12]|FY))?$")

Profile = Literal[
    "static",  # documents linked in the server-rendered HTML
    "static_url_filters",  # same, with listings filtered by URL parameters
    "no_direct_links",  # no document link in the start page HTML (JS, XHR or sub-pages)
    "js_xhr",  # confirmed: listings come from JavaScript / JSON responses
    "blocked",  # refuses the identifiable User-Agent (403, challenge)
    "unreachable",  # connection fails
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Site(_Model):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str
    category: str
    start_url: str
    profile: Profile
    qualified_on: dt.date
    recipe: str | None = None
    """File name in recipes/, when a hand-written (or validated) recipe exists."""
    snapshot: str | None = None
    """Snapshot directory relative to the repository root."""
    ground_truth: bool = False
    notes: str | None = None

    def snapshot_dir(self) -> Path | None:
        return ROOT / self.snapshot if self.snapshot else None

    def ground_truth_path(self) -> Path:
        return BENCH / "ground_truth" / f"{self.id}.yaml"


class SiteList(_Model):
    sites: list[Site]

    @model_validator(mode="after")
    def _unique(self) -> SiteList:
        ids = [s.id for s in self.sites]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate site ids")
        return self


class Document(_Model):
    url: str
    type: str
    period: str
    fiscal_year: int
    language: str = Field(pattern=r"^[a-z]{2,3}$")
    title: str | None = None
    date: dt.date | None = None

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in DOCUMENT_TYPES:
            raise ValueError(f"unknown document type {value!r} (see retrievr.taxonomy)")
        return value

    @field_validator("period", mode="before")
    @classmethod
    def _period(cls, value: object) -> str:
        text = str(value)
        if not PERIOD.match(text):
            raise ValueError(f"period {text!r} must look like 2025, 2025-Q3, 2025-H1, 2025-FY")
        return text

    @property
    def normalized_url(self) -> str:
        return normalize_url(self.url)


class Expectation(_Model):
    """Expected subset of the ground truth for a request, as a filter."""

    types: list[str] | None = None
    fiscal_years: list[int] | None = None
    languages: list[str] | None = None

    def matches(self, doc: Document) -> bool:
        return (
            (self.types is None or doc.type in self.types)
            and (self.fiscal_years is None or doc.fiscal_year in self.fiscal_years)
            and (self.languages is None or doc.language in self.languages)
        )


class Request(_Model):
    text: str = Field(min_length=5)
    expect: Expectation


class GroundTruth(_Model):
    ground_truth_version: Literal[1]
    site: str
    scope: str = Field(min_length=20)
    """Plain-language definition of what is in scope (pages, document kinds)."""
    fiscal_years: list[int] = Field(min_length=1)
    captured_at: dt.date
    reviewed: bool = False
    """Set to true by the human who checked every entry against the site."""
    notes: str | None = None
    documents: list[Document] = Field(min_length=1)
    requests: list[Request] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _consistent(self) -> GroundTruth:
        urls = [d.normalized_url for d in self.documents]
        if len(urls) != len(set(urls)):
            raise ValueError("duplicate document URLs")
        outside = [d.url for d in self.documents if d.fiscal_year not in self.fiscal_years]
        if outside:
            raise ValueError(f"documents outside fiscal_years: {outside[:3]}")
        for req in self.requests:
            if not any(req.expect.matches(d) for d in self.documents):
                raise ValueError(f"request {req.text!r} expects no document")
        return self

    def by_url(self) -> dict[str, Document]:
        return {d.normalized_url: d for d in self.documents}

    def expected_for(self, request: Request) -> set[str]:
        return {d.normalized_url for d in self.documents if request.expect.matches(d)}


def load_sites(path: Path = BENCH / "sites.yaml") -> SiteList:
    return SiteList.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_ground_truth(path: Path) -> GroundTruth:
    return GroundTruth.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
