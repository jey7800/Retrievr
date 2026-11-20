"""Recipe schema (pydantic, versioned) and its deterministic interpreter.

A recipe is data that describes a site: where the document listings are, how to
paginate, how to read items and how to type them. It never contains code; the
interpreter below is the only thing that acts on it.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retrievr.candidates import Candidate, compile_selector, extract_candidates, parse_html, select
from retrievr.classify import ClassifyRules, classify
from retrievr.fetch import BlockedError, Fetcher, FetchError, HTTPStatusError
from retrievr.urls import is_allowed, normalize_url, path_extension, resolve_url

log = logging.getLogger("retrievr.recipe")

SCHEMA_VERSION = 2
_SLUG = r"^[a-z0-9][a-z0-9-]*$"
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class YearRange(_Model):
    from_: int = Field(alias="from", ge=1900, le=2200)
    to: int | Literal["current"] = "current"

    def values(self, today: date) -> list[int]:
        end = today.year if self.to == "current" else self.to
        return list(range(self.from_, end + 1))


class NextLinkPagination(_Model):
    kind: Literal["next_link"] = "next_link"
    selector: str
    max_pages: int = Field(default=10, ge=1, le=200)

    @field_validator("selector")
    @classmethod
    def _valid_selector(cls, value: str) -> str:
        compile_selector(value)
        return value


class DateSpec(_Model):
    regex: str = Field(min_length=1, max_length=200)
    """Searched in the item text. Group 1 is used if present, else the whole match."""
    format: str = Field(min_length=2)
    """``datetime.strptime`` format, e.g. ``%d.%m.%Y``."""

    @field_validator("regex")
    @classmethod
    def _compiles(cls, value: str) -> str:
        re.compile(value)
        return value

    def parse(self, text: str) -> date | None:
        m = re.search(self.regex, text)
        if not m:
            return None
        raw = m.group(1) if m.groups() else m.group(0)
        try:
            return datetime.strptime(raw, self.format).date()
        except ValueError:
            return None


class ItemsSpec(_Model):
    container: str | None = None
    item: str | None = None
    link: str = "a[href]"
    title: str | None = None
    date: DateSpec | None = None
    extensions: list[str] | None = None
    table_context: bool = False
    """Schema v2: read row and column headers of links inside table cells."""

    @field_validator("container", "item", "link", "title")
    @classmethod
    def _valid_selector(cls, value: str | None) -> str | None:
        if value is not None:
            compile_selector(value)
        return value

    @field_validator("extensions")
    @classmethod
    def _lower_ext(cls, value: list[str] | None) -> list[str] | None:
        return [e.lower().lstrip(".") for e in value] if value is not None else None


class UrlTemplateSource(_Model):
    id: str = Field(pattern=_SLUG)
    kind: Literal["url_template"] = "url_template"
    url: str
    params: dict[str, YearRange] = {}
    pagination: NextLinkPagination | None = None
    items: ItemsSpec = ItemsSpec()

    @model_validator(mode="after")
    def _placeholders_match_params(self) -> UrlTemplateSource:
        used = set(_PLACEHOLDER.findall(self.url))
        declared = set(self.params)
        if used != declared:
            raise ValueError(
                f"source {self.id!r}: URL placeholders {sorted(used)} "
                f"do not match params {sorted(declared)}"
            )
        return self

    def expand(
        self, today: date, restrict: Mapping[str, Collection[int]] | None = None
    ) -> Iterator[tuple[dict[str, int], str]]:
        """Yield (params, url) for every parameter combination, in a stable order."""
        names = sorted(self.params)
        domains = []
        for name in names:
            values = self.params[name].values(today)
            if restrict and name in restrict:
                values = [v for v in values if v in set(restrict[name])]
            domains.append(values)
        for combo in itertools.product(*domains):
            params = dict(zip(names, combo, strict=True))
            yield params, _fill(self.url, params)


def _fill(template: str, params: Mapping[str, int]) -> str:
    return _PLACEHOLDER.sub(lambda m: str(params[m.group(1)]), template)


class FetchSpec(_Model):
    mode: Literal["http"] = "http"


class Defaults(_Model):
    language: str | None = Field(default=None, pattern=r"^[a-z]{2,3}$")


class Cadence(_Model):
    every: Literal["month", "quarter", "half_year", "year"]
    max_gap_days: int = Field(ge=1)


class HealthSpec(_Model):
    """Stored with the recipe; evaluated by health checks (later milestone)."""

    anchors: list[str] = []
    items_per_year: tuple[int, int] | None = None
    listing_fingerprint: str | None = None
    cadence: Cadence | None = None


class Provenance(_Model):
    method: Literal["hand_written", "learned", "repaired"]
    created_at: date | datetime | None = None
    notes: str | None = None


class Recipe(_Model):
    schema_version: Literal[1, 2]
    recipe: str = Field(pattern=_SLUG)
    version: int = Field(ge=1)
    issuer: str = Field(pattern=_SLUG)
    site: str
    allowed_domains: list[str] = Field(min_length=1)
    fetch: FetchSpec = FetchSpec()
    defaults: Defaults = Defaults()
    sources: list[UrlTemplateSource] = Field(min_length=1)
    classify: ClassifyRules = ClassifyRules()
    health: HealthSpec = HealthSpec()
    provenance: Provenance | None = None

    @field_validator("site")
    @classmethod
    def _lower_site(cls, value: str) -> str:
        return value.lower()

    @field_validator("allowed_domains")
    @classmethod
    def _lower_domains(cls, value: list[str]) -> list[str]:
        return [d.lower() for d in value]

    @model_validator(mode="after")
    def _consistent(self) -> Recipe:
        if self.site not in self.allowed_domains:
            raise ValueError(f"site {self.site!r} must be listed in allowed_domains")
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate source ids: {ids}")
        for source in self.sources:
            probe = _PLACEHOLDER.sub("2000", source.url)
            if not is_allowed(probe, self.allowed_domains):
                raise ValueError(f"source {source.id!r} URL is outside allowed_domains")
        if self.schema_version == 1:
            self._check_v1_subset()
        return self

    def _check_v1_subset(self) -> None:
        """Schema v1 recipes cannot use features added in v2."""
        if any(s.items.table_context for s in self.sources):
            raise ValueError("items.table_context requires schema_version 2")
        if not all(rule.uses_v1_targets_only for rule in self.classify.all_rules()):
            raise ValueError("rule targets other than url/text/any require schema_version 2")

    @property
    def key(self) -> str:
        return f"{self.recipe}@{self.version}"


def load_recipe(path: Path | str) -> Recipe:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Recipe.model_validate(data)


def recipe_from_mapping(data: Mapping[str, Any]) -> Recipe:
    return Recipe.model_validate(dict(data))


# -- interpreter -----------------------------------------------------------------


@dataclass(frozen=True)
class FoundDocument:
    url: str
    normalized_url: str
    title: str
    source_id: str
    params: Mapping[str, int]
    page_url: str
    extension: str
    date: date | None
    type: str | None
    period: str | None
    fiscal_year: int | None
    language: str | None
    score: float
    justification: str
    fields: Mapping[str, str | None] = field(default_factory=dict, compare=False)
    """Texts the classification rules read (link text, heading, table headers...)."""


@dataclass(frozen=True)
class PageVisit:
    source_id: str
    url: str
    final_url: str
    status: int
    items: int


RunStatus = Literal["ok", "blocked", "error"]


@dataclass
class RunResult:
    recipe: str
    recipe_version: int
    status: RunStatus = "ok"
    documents: list[FoundDocument] = field(default_factory=list)
    pages: list[PageVisit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_recipe(
    recipe: Recipe,
    fetcher: Fetcher,
    *,
    years: Collection[int] | None = None,
    today: date | None = None,
) -> RunResult:
    """Replay ``recipe``: enumerate and type every document it describes. No LLM call.

    ``years`` restricts the ``year`` parameter of url templates (when present).
    """
    today = today or date.today()
    result = RunResult(recipe=recipe.recipe, recipe_version=recipe.version)
    seen: set[str] = set()
    restrict = {"year": years} if years is not None else None
    for source in recipe.sources:
        for params, start_url in source.expand(today, restrict):
            try:
                _run_listing(recipe, source, params, start_url, fetcher, result, seen)
            except BlockedError as exc:
                result.status = "blocked"
                result.errors.append(str(exc))
                return result
            except HTTPStatusError as exc:
                if params and exc.status in (404, 410):
                    # A year page that does not exist (yet): nothing listed for it.
                    result.warnings.append(f"{source.id}: no listing for {params} ({exc})")
                    continue
                result.status = "error"
                result.errors.append(f"{source.id}: {exc}")
            except FetchError as exc:
                result.status = "error"
                result.errors.append(f"{source.id}: {exc}")
    return result


def _run_listing(
    recipe: Recipe,
    source: UrlTemplateSource,
    params: dict[str, int],
    start_url: str,
    fetcher: Fetcher,
    result: RunResult,
    seen: set[str],
) -> None:
    url: str | None = start_url
    visited: set[str] = set()
    max_pages = source.pagination.max_pages if source.pagination else 1
    while url is not None:
        key = normalize_url(url)
        if key in visited:
            break
        if len(visited) >= max_pages:
            result.warnings.append(f"{source.id}: stopped at max_pages={max_pages} ({url})")
            break
        visited.add(key)
        page = fetcher.get(url, allowed_domains=recipe.allowed_domains)
        root = parse_html(page.content)
        candidates = extract_candidates(
            root,
            page.final_url,
            container=source.items.container,
            item=source.items.item,
            link=source.items.link,
            title=source.items.title,
            extensions=source.items.extensions,
            table_context=source.items.table_context,
        )
        result.pages.append(PageVisit(source.id, url, page.final_url, page.status, len(candidates)))
        for cand in candidates:
            if not is_allowed(cand.url, recipe.allowed_domains):
                result.warnings.append(f"{source.id}: dropped link outside domains: {cand.url}")
                continue
            if cand.normalized_url in seen:
                continue
            seen.add(cand.normalized_url)
            result.documents.append(_to_document(recipe, source, params, page.final_url, cand))
        url = _next_url(root, source, page.final_url, recipe)


def _next_url(root: Any, source: UrlTemplateSource, base_url: str, recipe: Recipe) -> str | None:
    if source.pagination is None:
        return None
    for el in select(root, source.pagination.selector):
        href = el.get("href")
        if href:
            nxt = resolve_url(href, base_url)
            return nxt if is_allowed(nxt, recipe.allowed_domains) else None
    return None


def _to_document(
    recipe: Recipe,
    source: UrlTemplateSource,
    params: dict[str, int],
    page_url: str,
    cand: Candidate,
) -> FoundDocument:
    cls = classify(
        recipe.classify,
        url=cand.url,
        fields=cand.fields(),
        default_language=recipe.defaults.language,
    )
    doc_date = source.items.date.parse(cand.context_text) if source.items.date else None
    fiscal_year = cls.period_year or params.get("year") or (doc_date.year if doc_date else None)
    return FoundDocument(
        url=cand.url,
        normalized_url=cand.normalized_url,
        title=cand.title,
        source_id=source.id,
        params=dict(params),
        page_url=page_url,
        extension=path_extension(cand.url),
        date=doc_date,
        type=cls.type,
        period=cls.period,
        fiscal_year=fiscal_year,
        language=cls.language,
        score=cls.score,
        justification="; ".join(cls.justification),
        fields=cand.fields(),
    )
