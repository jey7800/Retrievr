"""Recipe schema (pydantic, versioned) and its deterministic interpreter.

A recipe is data that describes a site: where the document listings are, how to
paginate, how to read items and how to type them. It never contains code; the
interpreter below is the only thing that acts on it.

Schema versions: v1 (url templates, items, rules), v2 (table context, rule field
targets, ADR 0004), v3 (request sources: GET or POST endpoints with templated bodies,
values read from page options, page-number pagination, JSON or HTML responses, opaque
document links; ADR 0008).
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlencode

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from retrievr.candidates import (
    Candidate,
    compile_selector,
    extract_candidates,
    json_candidates,
    json_path,
    parse_html,
    select,
    text_of,
)
from retrievr.classify import ClassifyRules, classify
from retrievr.fetch import REQUEST_HEADERS, BlockedError, Fetcher, FetchError, HTTPStatusError
from retrievr.urls import is_allowed, normalize_url, resolve_url

log = logging.getLogger("retrievr.recipe")

SCHEMA_VERSION = 3
_SLUG = r"^[a-z0-9][a-z0-9-]*$"
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_TODAY = re.compile(r"\{today:([^{}]+)\}")
_YEAR_IN_TEXT = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
PAGE = "page"

ParamValue = int | str


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class YearRange(_Model):
    from_: int = Field(alias="from", ge=1900, le=2200)
    to: int | Literal["current"] = "current"

    def values(self, today: date) -> list[int]:
        end = today.year if self.to == "current" else self.to
        return list(range(self.from_, end + 1))


class OptionsParam(_Model):
    """Schema v3: values read from elements of a page, typically the ``<option>`` of a
    year selector whose values are opaque ids. The year is read in the element text."""

    kind: Literal["options"] = "options"
    page: str
    """Page holding the elements (GET, within the allowed domains)."""
    selector: str
    """CSS selector of the elements (e.g. ``#select-year option``)."""
    value_attr: str = Field(default="value", min_length=1)
    year: Literal["fiscal", "publication"] = "fiscal"
    """What the year of an option means. A ``publication`` bucket of year N also holds
    documents of fiscal year N-1 (annual reports published the next year)."""

    @field_validator("selector")
    @classmethod
    def _valid_selector(cls, value: str) -> str:
        compile_selector(value)
        return value


class NextLinkPagination(_Model):
    kind: Literal["next_link"] = "next_link"
    selector: str
    max_pages: int = Field(default=10, ge=1, le=200)

    @field_validator("selector")
    @classmethod
    def _valid_selector(cls, value: str) -> str:
        compile_selector(value)
        return value


class PageNumberPagination(_Model):
    """Schema v3: the ``{page}`` placeholder counts from ``start``. Stops at the first
    page with no new document, a 404, ``has_more`` false, or ``max_pages``."""

    kind: Literal["page_number"] = "page_number"
    start: int = Field(default=1, ge=0)
    max_pages: int = Field(default=50, ge=1, le=500)
    has_more: str | None = None
    """JSON path to a boolean telling whether another page exists (JSON responses)."""


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
    document_urls: list[str] | None = None
    """Schema v3: regular expressions; matching links are documents even without a
    file extension in their URL (opaque links)."""

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

    @field_validator("document_urls")
    @classmethod
    def _patterns(cls, value: list[str] | None) -> list[str] | None:
        for pattern in value or ():
            if len(pattern) > 300:
                raise ValueError("document_urls pattern longer than 300 characters")
            re.compile(pattern)
        return value


class JsonItems(_Model):
    """Schema v3: documents listed in a JSON response (dotted paths, list indexes as
    numbers)."""

    path: str
    """Path to the list of items, e.g. ``results.0.hits``."""
    url: str | list[str]
    """Path to the document URL inside an item; several paths: the first present wins."""
    title: str | None = None
    file_name: str | None = None
    date: str | None = None
    """Path to a date; its first ``YYYY-MM-DD`` is used."""
    context: list[str] = []
    """Paths of texts that describe the item (category, tags...), for the rules."""


class ResponseSpec(_Model):
    format: Literal["html", "json"] = "html"
    html_path: str | None = None
    """JSON response holding an HTML fragment at this path (read with ``items``)."""
    items: JsonItems | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ResponseSpec:
        if self.format == "json" and (self.html_path is None) == (self.items is None):
            raise ValueError("a json response needs exactly one of html_path or items")
        if self.format == "html" and (self.html_path or self.items):
            raise ValueError("html_path and items apply to json responses only")
        return self


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


class RequestSource(_Model):
    """Schema v3: a listing endpoint (often the XHR a page calls), replayed without a
    browser. Placeholders ``{name}`` (a param, ``year``, ``page``) and ``{today:%d/%m/%Y}``
    may appear in the URL, the form fields and the strings of the JSON body; a JSON
    string that is only ``{page}`` or ``{year}`` becomes a number."""

    id: str = Field(pattern=_SLUG)
    kind: Literal["request"] = "request"
    method: Literal["GET", "POST"] = "POST"
    url: str
    form: dict[str, str] | None = None
    json_body: Any = Field(default=None, alias="json")
    headers: dict[str, str] = {}
    params: dict[str, YearRange | OptionsParam] = {}
    pagination: PageNumberPagination | None = None
    response: ResponseSpec = ResponseSpec()
    items: ItemsSpec = ItemsSpec()

    @model_validator(mode="after")
    def _consistent(self) -> RequestSource:
        if self.form is not None and self.json_body is not None:
            raise ValueError(f"source {self.id!r}: give form or json, not both")
        if self.method == "GET" and (self.form is not None or self.json_body is not None):
            raise ValueError(f"source {self.id!r}: a GET request has no body")
        bad = [h for h in self.headers if h.lower() not in REQUEST_HEADERS]
        if bad:
            raise ValueError(f"source {self.id!r}: headers not allowed: {bad}")
        used = self.placeholders()
        allowed = set(self.params) | ({PAGE} if self.pagination else set())
        if any(isinstance(p, OptionsParam) for p in self.params.values()):
            allowed |= {"year", "publication_year"}
        if unknown := used - allowed:
            raise ValueError(f"source {self.id!r}: unknown placeholders {sorted(unknown)}")
        if self.pagination is not None and PAGE not in used:
            raise ValueError(f"source {self.id!r}: page_number pagination needs {{page}}")
        return self

    def placeholders(self) -> set[str]:
        texts = [self.url, *(self.form or {}).values(), *_strings(self.json_body)]
        return {name for text in texts for name in _PLACEHOLDER.findall(text)}

    def build(
        self, values: Mapping[str, ParamValue], today: date
    ) -> tuple[str, bytes | None, dict[str, str]]:
        """URL, body and headers of one request."""
        url = fill_template(self.url, values, today)
        headers = dict(self.headers)
        body: bytes | None = None
        if self.form is not None:
            form = {k: fill_template(v, values, today) for k, v in self.form.items()}
            body = urlencode(form).encode()
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
        elif self.json_body is not None:
            filled = _fill_json(self.json_body, values, today)
            body = json.dumps(filled, ensure_ascii=False, separators=(",", ":")).encode()
            headers.setdefault("Content-Type", "application/json")
        return url, body, headers


def _source_kind(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("kind", "url_template"))
    return str(getattr(value, "kind", "url_template"))


Source = Annotated[
    Annotated[UrlTemplateSource, Tag("url_template")] | Annotated[RequestSource, Tag("request")],
    Discriminator(_source_kind),
]
"""A source without ``kind`` is a ``url_template`` (schema v1 and v2 recipes)."""


def fill_template(text: str, values: Mapping[str, ParamValue], today: date) -> str:
    text = _TODAY.sub(lambda m: today.strftime(m.group(1)), text)
    return _PLACEHOLDER.sub(
        lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text
    )


def _fill(template: str, params: Mapping[str, int]) -> str:
    return _PLACEHOLDER.sub(lambda m: str(params[m.group(1)]), template)


def _fill_json(node: Any, values: Mapping[str, ParamValue], today: date) -> Any:
    if isinstance(node, dict):
        return {k: _fill_json(v, values, today) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill_json(v, values, today) for v in node]
    if isinstance(node, str):
        whole = _PLACEHOLDER.fullmatch(node)
        if whole and isinstance(values.get(whole.group(1)), int):
            return values[whole.group(1)]
        return fill_template(node, values, today)
    return node


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)
    elif isinstance(node, str):
        yield node


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
    schema_version: Literal[1, 2, 3]
    recipe: str = Field(pattern=_SLUG)
    version: int = Field(ge=1)
    issuer: str = Field(pattern=_SLUG)
    site: str
    allowed_domains: list[str] = Field(min_length=1)
    fetch: FetchSpec = FetchSpec()
    defaults: Defaults = Defaults()
    sources: list[Source] = Field(min_length=1)
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
            probe = _PLACEHOLDER.sub("2000", _TODAY.sub("x", source.url))
            if not is_allowed(probe, self.allowed_domains):
                raise ValueError(f"source {source.id!r} URL is outside allowed_domains")
            if isinstance(source, RequestSource):
                for param in source.params.values():
                    if isinstance(param, OptionsParam) and not is_allowed(
                        param.page, self.allowed_domains
                    ):
                        raise ValueError(f"source {source.id!r}: options page outside domains")
        if self.schema_version < 3:
            self._check_v2_subset()
        if self.schema_version == 1:
            self._check_v1_subset()
        return self

    def _check_v2_subset(self) -> None:
        """Schema v1 and v2 recipes cannot use features added in v3."""
        if any(isinstance(s, RequestSource) for s in self.sources):
            raise ValueError("request sources require schema_version 3")
        if any(s.items.document_urls for s in self.sources):
            raise ValueError("items.document_urls requires schema_version 3")
        if any(rule.period for rule in self.classify.period):
            raise ValueError("fixed period values require schema_version 3")
        if any(_targets(rule.target, "file_name") for rule in self.classify.all_rules()):
            raise ValueError("the file_name rule target requires schema_version 3")

    def _check_v1_subset(self) -> None:
        """Schema v1 recipes cannot use features added in v2."""
        if any(s.items.table_context for s in self.sources):
            raise ValueError("items.table_context requires schema_version 2")
        if not all(rule.uses_v1_targets_only for rule in self.classify.all_rules()):
            raise ValueError("rule targets other than url/text/any require schema_version 2")

    @property
    def key(self) -> str:
        return f"{self.recipe}@{self.version}"


def _targets(target: object, name: str) -> bool:
    return target == name or (isinstance(target, list) and name in target)


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
    params: Mapping[str, ParamValue]
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
    method: str = "GET"
    values: Mapping[str, ParamValue] = field(default_factory=dict)
    """Parameter values of a request source page (option, year, page)."""


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

    ``years`` restricts the ``year`` parameter of sources (when present).
    """
    today = today or date.today()
    result = RunResult(recipe=recipe.recipe, recipe_version=recipe.version)
    seen: set[str] = set()
    restrict = {"year": years} if years is not None else None
    for source in recipe.sources:
        try:
            if isinstance(source, RequestSource):
                _run_request_source(recipe, source, fetcher, result, seen, today, years)
                continue
            for params, start_url in source.expand(today, restrict):
                try:
                    _run_listing(recipe, source, params, start_url, fetcher, result, seen)
                except HTTPStatusError as exc:
                    if params and exc.status in (404, 410):
                        # A year page that does not exist (yet): nothing listed for it.
                        result.warnings.append(f"{source.id}: no listing for {params} ({exc})")
                        continue
                    raise
        except BlockedError as exc:
            result.status = "blocked"
            result.errors.append(str(exc))
            return result
        except FetchError as exc:
            result.status = "error"
            result.errors.append(f"{source.id}: {exc}")
        except ValueError as exc:  # unreadable response (bad JSON...)
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
        candidates = _html_candidates(root, page.final_url, source.items)
        result.pages.append(PageVisit(source.id, url, page.final_url, page.status, len(candidates)))
        _add(recipe, source.id, source.items, params, page.final_url, candidates, result, seen)
        url = _next_url(root, source, page.final_url, recipe)


def _html_candidates(root: Any, base_url: str, items: ItemsSpec) -> list[Candidate]:
    return extract_candidates(
        root,
        base_url,
        container=items.container,
        item=items.item,
        link=items.link,
        title=items.title,
        extensions=items.extensions,
        table_context=items.table_context,
        document_urls=items.document_urls,
    )


def _add(
    recipe: Recipe,
    source_id: str,
    items: ItemsSpec,
    params: Mapping[str, ParamValue],
    page_url: str,
    candidates: list[Candidate],
    result: RunResult,
    seen: set[str],
) -> int:
    """Add the new documents among ``candidates``; return how many were new."""
    added = 0
    for cand in candidates:
        if not is_allowed(cand.url, recipe.allowed_domains):
            result.warnings.append(f"{source_id}: dropped link outside domains: {cand.url}")
            continue
        if cand.normalized_url in seen:
            continue
        seen.add(cand.normalized_url)
        result.documents.append(_to_document(recipe, source_id, items, params, page_url, cand))
        added += 1
    return added


_Choice = tuple[str, ParamValue, int | None, bool]
"""(name, value, year, the year is a publication year)"""


def _assignments(
    recipe: Recipe,
    source: RequestSource,
    fetcher: Fetcher,
    today: date,
    years: Collection[int] | None,
) -> list[dict[str, ParamValue]]:
    """Every combination of parameter values. ``year`` is set when a parameter gives the
    fiscal year (it is the documents' fallback fiscal year), ``publication_year`` when it
    gives a publication year; ``years`` keeps the combinations that can hold them."""
    domains: list[list[_Choice]] = []
    for name in sorted(source.params):
        param = source.params[name]
        values: list[_Choice]
        if isinstance(param, YearRange):
            values = [(name, y, y if name == "year" else None, False) for y in param.values(today)]
        else:
            publication = param.year == "publication"
            values = [(name, v, y, publication) for v, y in _options(recipe, param, fetcher)]
        domains.append(values)
    out: list[dict[str, ParamValue]] = []
    for combo in itertools.product(*domains):
        values_by_name: dict[str, ParamValue] = {n: v for n, v, _y, _p in combo}
        keep = True
        for _n, _v, year, publication in combo:
            if year is None:
                continue
            # A bucket N+1 is read too: whatever the site calls the year, annual
            # documents of fiscal year N are often listed under the next year.
            keep &= years is None or year in years or year - 1 in years
            if publication:
                values_by_name.setdefault("publication_year", year)
            else:
                values_by_name.setdefault("year", year)
        if keep:
            out.append(values_by_name)
    return out


def _options(recipe: Recipe, param: OptionsParam, fetcher: Fetcher) -> list[tuple[str, int | None]]:
    page = fetcher.get(param.page, allowed_domains=recipe.allowed_domains)
    found = []
    for el in select(parse_html(page.content), param.selector):
        value = el.get(param.value_attr)
        if not value:
            continue
        m = _YEAR_IN_TEXT.search(text_of(el))
        found.append((value, int(m.group(1)) if m else None))
    return found


def _run_request_source(
    recipe: Recipe,
    source: RequestSource,
    fetcher: Fetcher,
    result: RunResult,
    seen: set[str],
    today: date,
    years: Collection[int] | None,
) -> None:
    for values in _assignments(recipe, source, fetcher, today, years):
        pagination = source.pagination
        pages: Iterator[int | None] = (
            iter(range(pagination.start, pagination.start + pagination.max_pages))
            if pagination
            else iter([None])
        )
        for page_number in pages:
            current = dict(values)
            if page_number is not None:
                current[PAGE] = page_number
            url, body, headers = source.build(current, today)
            try:
                page = fetcher.request(
                    source.method,
                    url,
                    content=body,
                    headers=headers,
                    allowed_domains=recipe.allowed_domains,
                )
            except HTTPStatusError as exc:
                first = pagination is None or page_number == pagination.start
                if exc.status in (404, 410) and (not first or values):
                    result.warnings.append(f"{source.id}: stop at {current} ({exc})")
                    break
                raise
            candidates, data = _response_candidates(source, page.content, page.final_url)
            result.pages.append(
                PageVisit(
                    source.id,
                    url,
                    page.final_url,
                    page.status,
                    len(candidates),
                    source.method,
                    dict(current),
                )
            )
            params = {k: v for k, v in current.items() if k != PAGE}
            new = _add(
                recipe, source.id, source.items, params, page.final_url, candidates, result, seen
            )
            if pagination is None or new == 0:
                break
            if pagination.has_more and not json_path(data, pagination.has_more):
                break
        else:
            if pagination is not None:
                result.warnings.append(f"{source.id}: stopped at max_pages={pagination.max_pages}")


def _response_candidates(
    source: RequestSource, content: bytes, base_url: str
) -> tuple[list[Candidate], Any]:
    spec = source.response
    if spec.format == "html":
        return _html_candidates(parse_html(content), base_url, source.items), None
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise ValueError(f"response of {base_url} is not JSON: {exc}") from exc
    if spec.html_path is not None:
        fragment = json_path(data, spec.html_path)
        html = fragment if isinstance(fragment, str) else ""
        root = parse_html(f"<html><body>{html}</body></html>")
        return _html_candidates(root, base_url, source.items), data
    assert spec.items is not None  # noqa: S101 - guaranteed by ResponseSpec
    # The date text joins the context, where _to_document reads its first ISO date.
    context = [*spec.items.context, *([spec.items.date] if spec.items.date else [])]
    found = json_candidates(
        data,
        base_url,
        items=spec.items.path,
        url=spec.items.url,
        title=spec.items.title,
        file_name=spec.items.file_name,
        context=context,
        extensions=source.items.extensions,
        document_urls=source.items.document_urls,
    )
    return found, data


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
    source_id: str,
    items: ItemsSpec,
    params: Mapping[str, ParamValue],
    page_url: str,
    cand: Candidate,
) -> FoundDocument:
    cls = classify(
        recipe.classify,
        url=cand.url,
        fields=cand.fields(),
        default_language=recipe.defaults.language,
    )
    doc_date = items.date.parse(cand.context_text) if items.date else None
    if doc_date is None and cand.dom_path.startswith("json:"):
        doc_date = _iso_date(cand.context_text)
    param_year = params.get("year")
    fiscal_year = (
        cls.period_year
        or (param_year if isinstance(param_year, int) else None)
        or (doc_date.year if doc_date else None)
    )
    return FoundDocument(
        url=cand.url,
        normalized_url=cand.normalized_url,
        title=cand.title,
        source_id=source_id,
        params=dict(params),
        page_url=page_url,
        extension=cand.extension,
        date=doc_date,
        type=cls.type,
        period=cls.period,
        fiscal_year=fiscal_year,
        language=cls.language,
        score=cls.score,
        justification="; ".join(cls.justification),
        fields=cand.fields(),
    )


def _iso_date(text: str) -> date | None:
    m = _ISO_DATE.search(text)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
