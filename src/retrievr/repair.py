"""Repair ladder for a broken recipe (rule 5: nothing is activated without validation).

1. **Relocation** (deterministic, no LLM): the documents the recipe listed before are
   looked for in the pages as they are now, and the parts of the recipe that stopped
   matching are rebuilt around them: the listing URL (a renamed query parameter), the
   year-selector options, the container, item and title selectors, the "next page" link.
   Each variant is replayed and kept only if it finds more known documents.
2. **Small model**: the cold-path learner runs again with a repair hint (the previous
   recipe, what the health check found, the documents to find again).
3. **Large model**: the same, with the large tier (skipped when it is the small model).
4. **Degraded**: the old recipe stays active and the site is marked ``degraded``.

Every candidate goes through the validation gates with the site's history: all anchors,
at least 95 % of the known documents still online, at most 1.5x the known volume, the
allowed domains, two identical runs, and the type and period of known documents kept.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Literal, cast
from urllib.parse import unquote, urlsplit

from lxml.html import HtmlElement
from pydantic import ValidationError

from retrievr.browser import Discoverer
from retrievr.candidates import (
    compile_selector,
    entry_element,
    extract_candidates,
    json_path,
    parse_html,
    select,
    text_of,
)
from retrievr.classify import Decision, apply_decisions
from retrievr.fetch import BlockedError, Fetcher, FetchError, Page
from retrievr.health import HealthReport, KnownDocument, Probe, in_scope, scope_years
from retrievr.learn import LearnContext, Learner, LearnError, RepairHint
from retrievr.llm import (
    BudgetExceededError,
    CostLedger,
    LLMError,
    Message,
    StructuredLLM,
    Tier,
)
from retrievr.recipe import (
    FoundDocument,
    OptionsParam,
    Provenance,
    Recipe,
    RequestSource,
    RunResult,
    Source,
    UrlTemplateSource,
    run_recipe,
)
from retrievr.urls import normalize_url, resolve_url
from retrievr.validate import History, ValidationReport, validate_runs

log = logging.getLogger("retrievr.repair")

RepairStatus = Literal["repaired", "degraded", "not_needed"]
MAX_PROBES = 10
_STABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_NEXT_TEXT = re.compile(
    r"^\s*(next|suivant|suivante|weiter|siguiente|successivo|›|»|>|load more|show more|"
    r"more results|voir plus|afficher plus|plus de r\S+sultats|mehr laden|mehr anzeigen)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RepairAttempt:
    rung: str
    """``relocate``, ``llm:small`` or ``llm:large``."""
    recipe: str | None
    passed: bool
    detail: str
    cost_usd: float = 0.0


@dataclass
class RepairResult:
    site: str
    status: RepairStatus
    recipe: Recipe | None = None
    """The repaired recipe (validated, not yet saved), when ``repaired``."""
    documents: list[FoundDocument] = field(default_factory=list)
    """Its documents, stored decisions applied."""
    decisions: dict[str, Decision] = field(default_factory=dict)
    """Decisions to store with it (known documents keep their metadata)."""
    validation: ValidationReport | None = None
    attempts: list[RepairAttempt] = field(default_factory=list)
    reason: str = ""


class MemoFetcher:
    """A fetcher that answers each identical request once (relocation replays the same
    pages many times). Errors are remembered too."""

    def __init__(self, inner: Fetcher) -> None:
        self.inner = inner
        self._memo: dict[tuple[str, str, bytes | None], Page | FetchError] = {}

    def get(self, url: str, **kwargs: Any) -> Page:
        return self.request("GET", url, **kwargs)

    def request(
        self, method: str, url: str, *, content: bytes | None = None, **kwargs: Any
    ) -> Page:
        key = (method, url, content)
        if key not in self._memo:
            try:
                self._memo[key] = self.inner.request(
                    cast(Any, method), url, content=content, **kwargs
                )
            except BlockedError:
                raise
            except FetchError as exc:
                self._memo[key] = exc
        found = self._memo[key]
        if isinstance(found, FetchError):
            raise found
        return found

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _TierLLM:
    """Routes every call of the learner to one model tier."""

    def __init__(self, inner: StructuredLLM, tier: Tier) -> None:
        self.inner = inner
        self.tier = tier

    def structured(
        self,
        messages: Sequence[Message],
        schema: type[Any],
        *,
        step: str,
        tier: Tier = "small",
        model: str | None = None,
    ) -> Any:
        return self.inner.structured(
            messages, schema, step=f"repair:{step}", tier=self.tier, model=model
        )


@dataclass
class RepairContext:
    site: str
    today: date
    years: Collection[int] | None
    version: int
    """Version number of the repaired recipe."""
    start_url: str | None = None
    decisions: dict[str, Decision] = field(default_factory=dict)
    llm: StructuredLLM | None = None
    tiers: Sequence[Tier] = ("small", "large")
    llm_attempts_left: int = 2
    ledger: CostLedger | None = None
    discoverer: Discoverer | None = None
    probe: Probe | None = None


class Repairer:
    def __init__(
        self,
        recipe: Recipe,
        report: HealthReport,
        known: Sequence[KnownDocument],
        fetcher: Fetcher,
        ctx: RepairContext,
    ) -> None:
        self.recipe = recipe
        self.report = report
        self.fetcher = fetcher
        self.ctx = ctx
        self.scope = scope_years(recipe, ctx.years)
        removed = set(report.removed)
        self.known_all = {k.normalized_url for k in known}
        self.targets = [
            k for k in known if in_scope(k, self.scope) and k.normalized_url not in removed
        ]

    # -- ladder ------------------------------------------------------------------------

    def repair(self) -> RepairResult:
        result = RepairResult(site=self.ctx.site, status="degraded")
        if not self.targets:
            result.reason = "no known document to check a repair against"
            return result
        memo = MemoFetcher(self.fetcher)
        relocated = Relocator(self.recipe, cast(Fetcher, memo), self).relocate()
        if relocated is not None and self._try("relocate", relocated, result):
            return result
        if relocated is None:
            result.attempts.append(RepairAttempt("relocate", None, False, "nothing relocated"))
        for tier in self._tiers():
            if self.ctx.llm_attempts_left <= 0:
                result.reason = "daily limit of LLM repair attempts reached"
                break
            self.ctx.llm_attempts_left -= 1
            spent = self._spent()
            try:
                candidate = self._learn(tier)
            except BudgetExceededError as exc:
                result.attempts.append(RepairAttempt(f"llm:{tier}", None, False, str(exc)))
                result.reason = f"budget exceeded: {exc}"
                break
            except (LearnError, LLMError, ValueError, ValidationError) as exc:
                detail = f"{type(exc).__name__}: {exc}"[:300]
                result.attempts.append(
                    RepairAttempt(f"llm:{tier}", None, False, detail, self._spent() - spent)
                )
                continue
            if self._try(f"llm:{tier}", candidate, result, cost=self._spent() - spent):
                return result
        if not result.reason:
            result.reason = "no candidate passed validation"
        return result

    def _tiers(self) -> list[Tier]:
        if self.ctx.llm is None:
            return []
        tiers = list(self.ctx.tiers)
        settings = getattr(self.ctx.llm, "settings", None)
        if settings is not None and settings.model_large in (None, "", settings.model_small):
            tiers = [t for t in tiers if t != "large"]
        return tiers

    def _spent(self) -> float:
        return self.ctx.ledger.spent_usd if self.ctx.ledger is not None else 0.0

    def _learn(self, tier: Tier) -> Recipe:
        if self.ctx.llm is None:
            raise LearnError("no LLM configured")
        start = self.ctx.start_url or default_start_url(self.recipe, self.ctx.today, self.targets)
        context = LearnContext(
            issuer=self.recipe.issuer, start_url=start, years=self.ctx.years, today=self.ctx.today
        )
        hint = RepairHint(
            previous=self.recipe,
            failures=[f"{c.name}: {c.detail}" for c in self.report.failures()],
            known=[(k.normalized_url, k.title) for k in self.targets],
        )
        learner = Learner(
            self.fetcher,
            _TierLLM(self.ctx.llm, tier),
            context,
            discoverer=self.ctx.discoverer,
            hint=hint,
        )
        structure = learner.learn_structure()
        built = learner.build_recipe(
            structure.reply, rules=self.recipe.classify, health=self.recipe.health
        )
        return self.as_repair(built, f"{tier} model: {structure.reply.reasoning[:200]}")

    def as_repair(self, candidate: Recipe, notes: str) -> Recipe:
        """``candidate`` as the next version of the broken recipe (name, rules, health and
        domains carried over)."""
        data = candidate.model_dump(mode="json", by_alias=True, exclude_none=True)
        data.update(
            recipe=self.recipe.recipe,
            issuer=self.recipe.issuer,
            version=self.ctx.version,
            allowed_domains=sorted(
                set(candidate.allowed_domains) | set(self.recipe.allowed_domains)
            ),
            classify=self.recipe.classify.model_dump(mode="json", exclude_none=True),
            health=self.recipe.health.model_dump(mode="json", exclude_none=True),
            provenance=Provenance(
                method="repaired", created_at=self.ctx.today, notes=notes[:300]
            ).model_dump(mode="json", exclude_none=True),
        )
        if candidate.defaults.language is None:
            data["defaults"] = self.recipe.defaults.model_dump(mode="json", exclude_none=True)
        return Recipe.model_validate(data)

    # -- validation --------------------------------------------------------------------

    def decisions(self) -> dict[str, Decision]:
        """Stored decisions plus the metadata of known documents (rules still win)."""
        out = dict(self.ctx.decisions)
        for k in self.targets:
            if k.type is not None or k.period is not None:
                out.setdefault(
                    k.normalized_url,
                    Decision(k.type, k.period, k.fiscal_year, k.language, "history"),
                )
        return out

    def validate(
        self, candidate: Recipe
    ) -> tuple[ValidationReport, RunResult, dict[str, Decision]]:
        first = run_recipe(candidate, self.fetcher, years=self.ctx.years, today=self.ctx.today)
        decisions = self.decisions()
        decided = replace(first, documents=apply_decisions(first.documents, decisions))
        again = run_recipe(candidate, self.fetcher, years=self.ctx.years, today=self.ctx.today)
        second = replace(again, documents=apply_decisions(again.documents, decisions))
        found = {d.normalized_url for d in decided.documents}
        targets = {k.normalized_url for k in self.targets}
        gone = self._gone(sorted(targets - found))
        online = targets - gone
        anchors = {normalize_url(a) for a in self.recipe.health.anchors}
        anchors = {a for a in anchors if a in online or (self.ctx.years is None and a not in gone)}
        scope = self.scope
        outside = self.known_all - {k.normalized_url for k in self.targets}

        def counted(doc: FoundDocument) -> bool:
            # Known documents of other years do not count, even when the candidate cannot
            # tell their year.
            return doc.normalized_url not in outside and in_scope(KnownDocument.of(doc), scope)

        history = History(
            anchors=sorted(anchors),
            known_online=sorted(online),
            volume=len(self.targets),
            scope=counted,
            metadata={
                k.normalized_url: (k.type, k.period)
                for k in self.targets
                if k.normalized_url in online and (k.type is not None or k.period is not None)
            },
        )
        return validate_runs(candidate, decided, second, history=history), decided, decisions

    def _gone(self, missing: Sequence[str]) -> set[str]:
        if self.ctx.probe is None:
            return set()
        return {u for u in missing[:MAX_PROBES] if self.ctx.probe(u) is False}

    def _try(self, rung: str, candidate: Recipe, result: RepairResult, cost: float = 0.0) -> bool:
        report, run, decisions = self.validate(candidate)
        result.attempts.append(
            RepairAttempt(rung, candidate.key, report.passed, report.summary(), cost)
        )
        log.info("repair %s %s: %s", self.ctx.site, rung, report.summary())
        log.debug(
            "candidate sources: %s",
            [x.model_dump(mode="json", exclude_none=True) for x in candidate.sources],
        )
        if not report.passed:
            return False
        result.status = "repaired"
        result.recipe = candidate
        result.documents = run.documents
        result.decisions = decisions
        result.validation = report
        result.reason = f"repaired by {rung}"
        return True


def default_start_url(recipe: Recipe, today: date, targets: Sequence[KnownDocument]) -> str:
    """An entry page for a recipe learned elsewhere: the page of its first source (for the
    year that lists the most known documents), or the options page of a request source."""
    years = sorted((k.listed_year for k in targets if k.listed_year), key=lambda y: -y)
    for source in recipe.sources:
        if isinstance(source, UrlTemplateSource):
            combos = list(source.expand(today))
            if years:
                best = [c for c in combos if c[0].get("year") == years[0]]
                combos = best or combos
            if combos:
                return combos[-1][1]
        else:
            for param in source.params.values():
                if isinstance(param, OptionsParam):
                    return param.page
    return f"https://{recipe.site}/"


# -- relocation ------------------------------------------------------------------------


class Relocator:
    """Rebuild the parts of a recipe that stopped matching, around the known documents."""

    def __init__(self, recipe: Recipe, fetcher: Fetcher, repairer: Repairer) -> None:
        self.recipe = recipe
        self.fetcher = fetcher
        self.repairer = repairer
        self.today = repairer.ctx.today
        self.years = repairer.ctx.years
        self.targets = {k.normalized_url for k in repairer.targets}
        self.known_all = repairer.known_all
        self.dead = False
        """Set by a variant step when the part it rebuilds matches nothing at all."""
        self.by_source: dict[str, list[KnownDocument]] = {}
        for k in repairer.targets:
            self.by_source.setdefault(k.source_id, []).append(k)

    def relocate(self) -> Recipe | None:
        best = self.recipe
        best_score = self.score(best)
        steps: list[Callable[[Recipe, int], Iterator[Recipe]]] = [
            self._url_variants,
            self._options_variants,
            self._items_variants,
            self._pagination_variants,
        ]
        for _pass in range(2):  # a second pass when parts depend on each other
            changed = False
            for index in range(len(self.recipe.sources)):
                for step in steps:
                    self.dead = False
                    variants = list(step(best, index))
                    if not variants:
                        continue
                    scored = [(self.score(v), v) for v in variants]
                    top = max(score for score, _v in scored)
                    chosen = next(v for score, v in scored if score == top)
                    # A part that matches nothing at all is replaced even when the
                    # recipe finds nothing yet (another part is still broken).
                    if top > best_score or (self.dead and top == best_score):
                        best, best_score, changed = chosen, top, True
            if not changed:
                break
        if best is self.recipe:
            return None
        log.info("relocation of %s: recall %.0f%%", self.recipe.key, 100 * best_score[0])
        return self.repairer.as_repair(best, "relocated: selectors rebuilt around known documents")

    def score(self, recipe: Recipe) -> tuple[float, int]:
        run = run_recipe(recipe, self.fetcher, years=self.years, today=self.today)
        found = {d.normalized_url for d in run.documents}
        recall = len(found & self.targets) / len(self.targets) if self.targets else 0.0
        return round(recall, 4), -len(found - self.known_all)

    # -- helpers -----------------------------------------------------------------------

    def _known_for(self, source: Source) -> list[KnownDocument]:
        return self.by_source.get(source.id) or list(self.repairer.targets)

    def _fetch(self, url: str) -> Page | None:
        try:
            return self.fetcher.get(url, allowed_domains=self.recipe.allowed_domains)
        except BlockedError:
            raise
        except FetchError:
            return None

    def _listing_url(self, source: UrlTemplateSource) -> str | None:
        combos = list(source.expand(self.today))
        if not combos:
            return None
        years = [k.listed_year for k in self._known_for(source) if k.listed_year]
        if years and source.params:
            common = max(set(years), key=lambda y: (years.count(y), y))
            chosen = [c for c in combos if c[0].get("year") == common]
            combos = chosen or combos
        return combos[-1][1]

    def _listing_root(self, source: Source) -> tuple[HtmlElement, str] | None:
        if isinstance(source, UrlTemplateSource):
            url = self._listing_url(source)
            page = self._fetch(url) if url else None
            return (parse_html(page.content), page.final_url) if page else None
        return self._request_root(source)

    def _request_root(self, source: RequestSource) -> tuple[HtmlElement, str] | None:
        if source.response.format == "json" and source.response.html_path is None:
            return None  # a JSON list: no HTML to relocate in
        known = [k for k in self._known_for(source) if k.params]
        if not known:
            return None
        values = dict(known[-1].params)
        if source.pagination is not None:
            values["page"] = source.pagination.start
        url, body, headers = source.build(values, self.today)
        try:
            page = self.fetcher.request(
                source.method,
                url,
                content=body,
                headers=headers,
                allowed_domains=self.recipe.allowed_domains,
            )
        except BlockedError:
            raise
        except FetchError:
            return None
        if source.response.html_path is None:
            return parse_html(page.content), page.final_url
        try:
            fragment = json_path(json.loads(page.content), source.response.html_path)
        except ValueError:
            return None
        if not isinstance(fragment, str) or not fragment.strip():
            return None
        return parse_html(fragment), page.final_url

    def _known_links(self, root: HtmlElement, base: str) -> list[HtmlElement]:
        out = []
        for a in root.iter("a"):
            href = a.get("href")
            if href and normalize_url(resolve_url(href, base)) in self.known_all:
                out.append(a)
        return out

    def _with_source(self, recipe: Recipe, index: int, source: Source) -> Recipe | None:
        data = recipe.model_dump(mode="json", by_alias=True, exclude_none=True)
        data["sources"][index] = source.model_dump(mode="json", by_alias=True, exclude_none=True)
        try:
            return Recipe.model_validate(data)
        except ValidationError:
            return None

    # -- variants ----------------------------------------------------------------------

    def _url_variants(self, recipe: Recipe, index: int) -> Iterator[Recipe]:
        """A renamed query parameter: the page (or the start page) links to the same
        listing with the year in another parameter."""
        source = recipe.sources[index]
        if not isinstance(source, UrlTemplateSource) or "{year}" not in source.url:
            return
        path, _, query = source.url.partition("?")
        segments = query.split("&") if query else []
        year_at = next((i for i, s in enumerate(segments) if "{year}" in s), None)
        if year_at is None:
            return
        others = {_segment(s) for i, s in enumerate(segments) if i != year_at}
        pages = [self._listing_url(source), self.repairer.ctx.start_url]
        seen: set[str] = set()
        for url in [u for u in pages if u]:
            page = self._fetch(url)
            if page is None:
                continue
            root = parse_html(page.content)
            for a in root.iter("a"):
                href = a.get("href")
                if not href:
                    continue
                link_path, _, link_query = resolve_url(href, page.final_url).partition("?")
                if link_path != path or not link_query:
                    continue
                link_segments = link_query.split("&")
                for seg in link_segments:
                    if not _YEAR.search(seg) or _segment(seg) in others:
                        continue
                    rest = {_segment(s) for s in link_segments if s != seg}
                    if not others <= rest:
                        continue
                    templated = _YEAR.sub("{year}", seg, count=1)
                    if _segment(templated) == _segment(segments[year_at]):
                        continue
                    new_segments = list(segments)
                    new_segments[year_at] = templated
                    new_url = f"{path}?{'&'.join(new_segments)}"
                    if new_url in seen:
                        continue
                    seen.add(new_url)
                    variant = self._with_source(
                        recipe, index, source.model_copy(update={"url": new_url})
                    )
                    if variant is not None:
                        yield variant

    def _options_variants(self, recipe: Recipe, index: int) -> Iterator[Recipe]:
        """Renamed year selectors: the ``<select>`` whose option values are those the
        known documents were listed under."""
        source = recipe.sources[index]
        if not isinstance(source, RequestSource):
            return
        for name, param in source.params.items():
            if not isinstance(param, OptionsParam):
                continue
            page = self._fetch(param.page)
            if page is None:
                continue
            root = parse_html(page.content)
            if select(root, param.selector):
                continue
            self.dead = True
            values = {str(k.params[name]) for k in self._known_for(source) if name in k.params}
            if not values:
                continue
            for sel in root.iter("select"):
                offered = {o.get(param.value_attr) for o in sel.iter("option")}
                if len(values & offered) < 0.5 * len(values):
                    continue
                selector = _element_selector(sel)
                if selector is None:
                    continue
                new_param = param.model_copy(update={"selector": f"{selector} option"})
                params = dict(source.params)
                params[name] = new_param
                variant = self._with_source(
                    recipe, index, source.model_copy(update={"params": params})
                )
                if variant is not None:
                    yield variant

    def _items_variants(self, recipe: Recipe, index: int) -> Iterator[Recipe]:
        """Renamed classes or ids, an extra wrapper: container, item, title and link
        selectors rebuilt from the elements that hold known documents."""
        source = recipe.sources[index]
        found = self._listing_root(source)
        if found is None:
            return
        root, base = found
        links = self._known_links(root, base)
        if len(links) < 2:
            return
        old = source.items
        listed = extract_candidates(
            root,
            base,
            container=old.container,
            item=old.item,
            link=old.link,
            extensions=old.extensions,
            document_urls=old.document_urls,
        )
        self.dead = not any(c.normalized_url in self.known_all for c in listed)
        items = _items_of(links, old.item, base)
        item_selectors = [_distinctive(items, root), _compound(items)] if old.item else [None]
        containers = _container_selectors(items or links, root)
        titles: list[str | None] = [old.title]
        if old.title and not _plain_tag(old.title):
            titles.append(self._title_selector(items, source))
        link_selectors = [old.link]
        if old.link != "a[href]":
            link_selectors.append("a[href]")
        variants = []
        for container in _unique([old.container, *containers]):
            for item in _unique([old.item, *item_selectors]):
                for title in _unique(titles):
                    for link in _unique(link_selectors):
                        variants.append((container, item, title, link))
        for container, item, title, link in variants[1:]:  # the first is the old spec
            spec = old.model_copy(
                update={"container": container, "item": item, "title": title, "link": link}
            )
            variant = self._with_source(recipe, index, source.model_copy(update={"items": spec}))
            if variant is not None:
                yield variant

    def _title_selector(self, items: Sequence[HtmlElement], source: Source) -> str | None:
        titles = {" ".join(k.title.split()) for k in self._known_for(source) if k.title}
        holders = []
        for item in items:
            for el in item.iter():
                text = " ".join(text_of(el).split()) if isinstance(el.tag, str) else ""
                if el.tag != "a" and text in titles:
                    holders.append(el)
                    break
        return _compound(holders) if holders else None

    def _pagination_variants(self, recipe: Recipe, index: int) -> Iterator[Recipe]:
        """A "next page" link that changed: rel=next, a link to the next page number, or a
        "next" / "load more" link."""
        source = recipe.sources[index]
        if not isinstance(source, UrlTemplateSource) or source.pagination is None:
            return
        url = self._listing_url(source)
        page = self._fetch(url) if url else None
        if page is None:
            return
        root = parse_html(page.content)
        if select(root, source.pagination.selector):
            return
        selectors: list[str] = []
        if select(root, "a[rel=next]"):
            selectors.append("a[rel=next]")
        for a in root.iter("a"):
            href = a.get("href")
            if not href:
                continue
            if _next_page_of(page.final_url, resolve_url(href, page.final_url)) or _NEXT_TEXT.match(
                text_of(a)
            ):
                selector = _link_selector(a)
                if selector:
                    selectors.append(selector)
        for selector in _unique(selectors):
            pagination = source.pagination.model_copy(update={"selector": selector})
            variant = self._with_source(
                recipe, index, source.model_copy(update={"pagination": pagination})
            )
            if variant is not None:
                yield variant


# -- selector building -----------------------------------------------------------------


def _unique(values: Sequence[str | None]) -> list[str | None]:
    out: list[str | None] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def _stable_classes(el: HtmlElement) -> set[str]:
    return {
        c
        for c in (el.get("class") or "").split()
        if _STABLE_NAME.match(c) and not re.search(r"\d{3,}", c)
    }


def _stable_id(el: HtmlElement) -> str | None:
    value = el.get("id")
    if value and _STABLE_NAME.match(value) and not _YEAR.search(value):
        return str(value)
    return None


def _compound(elements: Sequence[HtmlElement]) -> str | None:
    """``tag.class...`` matching every element (classes they all share)."""
    if not elements:
        return None
    tags = {el.tag for el in elements}
    classes = set.intersection(*(_stable_classes(el) for el in elements))
    if len(tags) != 1 and not classes:
        return None
    tag = next(iter(tags)) if len(tags) == 1 else ""
    selector = str(tag) + "".join(f".{c}" for c in sorted(classes))
    return selector if _valid(selector) else None


def _valid(selector: str) -> bool:
    try:
        compile_selector(selector)
    except ValueError:
        return False
    return bool(selector)


def _plain_tag(selector: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9]*", selector.strip()))


def _element_selector(el: HtmlElement) -> str | None:
    if (value := _stable_id(el)) is not None:
        return f"#{value}"
    name = el.get("name")
    if name and _STABLE_NAME.match(name):
        return f'{el.tag}[name="{name}"]'
    classes = _stable_classes(el)
    return f"{el.tag}" + "".join(f".{c}" for c in sorted(classes)) if classes else None


def _link_selector(a: HtmlElement) -> str | None:
    if _stable_classes(a):
        return _compound([a])
    rel = a.get("rel")
    if rel and _STABLE_NAME.match(rel):
        return f"a[rel={rel}]"
    parent = a.getparent()
    if parent is not None and (holder := _element_selector(parent)) is not None:
        return f"{holder} a"
    return None


def _item_tag(selector: str | None) -> str | None:
    if not selector:
        return None
    last = re.split(r"[\s>+~]+", selector.strip())[-1]
    m = re.match(r"[a-zA-Z][a-zA-Z0-9]*", last)
    return m.group(0).lower() if m else None


def _items_of(links: Sequence[HtmlElement], old_item: str | None, base: str) -> list[HtmlElement]:
    """The entry element of each link: its nearest ancestor with the tag the old item
    selector named, else its row."""
    tag = _item_tag(old_item)
    items: list[HtmlElement] = []
    for a in links:
        holder = None
        if tag:
            holder = next((el for el in a.iterancestors() if el.tag == tag), None)
        holder = holder if holder is not None else entry_element(a, base)
        if holder not in items:
            items.append(holder)
    return items


def _lowest_common_ancestor(elements: Sequence[HtmlElement]) -> HtmlElement | None:
    chains = [[*reversed(list(el.iterancestors())), el] for el in elements]
    common = None
    for nodes in zip(*chains, strict=False):
        if all(n is nodes[0] for n in nodes):
            common = nodes[0]
        else:
            break
    return common


def _container_selectors(elements: Sequence[HtmlElement], root: HtmlElement) -> list[str]:
    """Selectors of the element enclosing ``elements``, the most stable first: an id
    (of it or an ancestor), then its most distinctive class, then all its classes."""
    lca = _lowest_common_ancestor(elements)
    if lca is None:
        return []
    if len(elements) == 1:
        lca = lca.getparent() if lca.getparent() is not None else lca
    out: list[str] = []
    for node in [lca, *lca.iterancestors()]:
        if node.tag in ("body", "html"):
            break
        if (value := _stable_id(node)) is not None:
            out.append(f"#{value}")
            break
    if _stable_classes(lca):
        out += [s for s in (_distinctive([lca], root), _compound([lca])) if s]
    return out


def _distinctive(elements: Sequence[HtmlElement], root: HtmlElement) -> str | None:
    """``tag.class`` with one class that selects exactly ``elements`` in the page: the
    first such class in the elements' own order (authors put the meaningful one first,
    before layout utilities)."""
    if not elements:
        return None
    tags = {el.tag for el in elements}
    if len(tags) != 1:
        return None
    tag = str(next(iter(tags)))
    shared = set.intersection(*(_stable_classes(el) for el in elements))
    wanted = set(elements)
    order = (elements[0].get("class") or "").split()
    for name in [c for c in order if c in shared]:
        selector = f"{tag}.{name}"
        if _valid(selector) and set(select(root, selector)) == wanted:
            return selector
    return None


def _next_page_of(current: str, target: str) -> bool:
    """Whether ``target`` is ``current`` with a page number one higher (or a first page
    number added)."""
    a, b = urlsplit(current), urlsplit(target)
    if (a.netloc, a.path) != (b.netloc, b.path):
        return False
    qa = _query(a.query)
    qb = _query(b.query)
    changed = [k for k in qb if qa.get(k) != qb[k]]
    if len(changed) != 1 or any(k not in qb for k in qa):
        return False
    key = changed[0]
    if not qb[key].isdigit():
        return False
    before = qa.get(key)
    return (before is None and int(qb[key]) in (1, 2)) or (
        before is not None and before.isdigit() and int(qb[key]) == int(before) + 1
    )


def _query(query: str) -> dict[str, str]:
    out = {}
    for part in query.split("&"):
        if part:
            key, _, value = part.partition("=")
            out[unquote(key)] = unquote(value)
    return out


def _segment(segment: str) -> str:
    """A query segment compared loosely: decoded, ``key=`` same as ``key``."""
    return unquote(segment).removesuffix("=")
