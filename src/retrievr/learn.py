"""Cold path: learn a recipe for a new site, and compile a request into a filter.

The LLM never writes code. It fills small JSON objects (pydantic schemas below) that the
engine turns into a :class:`~retrievr.recipe.Recipe`, then replays to check them:

1. **Structure** (bounded loop): the LLM sees a digest of the pages (document link groups,
   navigation links, pruned HTML). It may ask to open up to 5 listed links (twice at
   most), then proposes the listing sources. The engine runs them and compares with the
   groups the LLM selected; a diagnostic goes back for at most two repairs.
2. **Rules**: the LLM writes the type, period and language rules and labels a sample of
   documents. The engine measures how well the rules reproduce the labels and asks for
   one repair if they do not.
3. **Request**: the natural-language request becomes a :class:`Filter`, applied without
   the LLM.

Page content is wrapped as untrusted data in every prompt, and every URL a recipe holds
must stay within its allowed domains.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from retrievr.candidates import (
    DOCUMENT_EXTENSIONS,
    LinkGroup,
    NavLink,
    entry_element,
    group_document_links,
    in_site,
    navigation_links,
    page_title,
    parse_html,
    pruned_html,
    site_domain,
)
from retrievr.classify import (
    ClassifyRules,
    Decision,
    DocumentLabel,
    LanguageRule,
    PeriodRule,
    TypeRule,
    apply_decisions,
    classify,
    classify_with_llm,
    document_table,
    needs_decision,
)
from retrievr.fetch import BlockedError, Fetcher, FetchError
from retrievr.llm import (
    UNTRUSTED_NOTICE,
    Message,
    StructuredLLM,
    StructuredOutputError,
    untrusted,
)
from retrievr.recipe import (
    SCHEMA_VERSION,
    Cadence,
    Defaults,
    FoundDocument,
    HealthSpec,
    ItemsSpec,
    NextLinkPagination,
    Provenance,
    Recipe,
    RunResult,
    UrlTemplateSource,
    YearRange,
    run_recipe,
)
from retrievr.taxonomy import DocumentType, taxonomy_text
from retrievr.urls import host_of, is_allowed, normalize_url

log = logging.getLogger("retrievr.learn")

M = TypeVar("M", bound=BaseModel)
MAX_GROUPS_PER_PAGE = 30
MAX_SNIPPETS_PER_PAGE = 12
MAX_NAV_PER_PAGE = 60
MAX_OPEN_PER_ROUND = 5
RULES_AGREEMENT = 0.9
MIN_RULES_SAMPLE = 30
_YEAR_PLACEHOLDER = "{year}"


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- structure step ------------------------------------------------------------------


class SourceDraft(_Reply):
    url: str
    """Exact URL of a listing page; the year may be replaced by {year}."""
    year_from: int | None
    """Earliest year available, when the URL contains {year}."""
    container: str | None
    item: str | None
    title: str | None
    pagination_next_selector: str | None
    table_context: bool


class StructureReply(_Reply):
    reasoning: str
    action: Literal["open_pages", "recipe"]
    open_links: list[str]
    """Navigation link ids (``P1.N4``) to open, for action ``open_pages``."""
    document_groups: list[str]
    """Group ids (``P1.G2``) holding the documents in scope, for action ``recipe``."""
    sources: list[SourceDraft]
    default_language: str | None


STRUCTURE_SYSTEM = f"""You configure retrievr, a deterministic engine that lists the
financial documents of investor-relations web pages. You never write code: you fill a
JSON object that the engine interprets and checks.

Goal: describe where the site lists its documents (reports, results presentations, press
releases, financial data, transcripts, registration documents...) so that the engine can
find ALL of them for every fiscal year, now and in future years, with no help from you.
{UNTRUSTED_NOTICE}

You get a digest of each fetched page: groups of document links (links to files that
share a DOM path; digits are folded, so per-year panels form one group) and navigation
links to other pages of the site.

Choose an action:
1. "open_pages": the listing of past years (or the full listing) is on pages you have not
   seen yet: an archive, per-year pages, an "all reports" view, a bigger page size. List up
   to 5 navigation link ids in open_links. Do not open pages you do not need.
2. "recipe": describe the listing.
   - document_groups: ids of the groups that are the document listing in scope. Leave out
     navigation menus, banners, footers, legal or cookie documents, and documents repeated
     on every page.
   - sources: one per listing page. When the listing exists per year and the page URLs
     differ only by the year, use ONE source whose url contains the placeholder {{year}}
     and set year_from to the earliest year offered. A page that already holds every year
     needs no placeholder (year_from null). Copy the URL of a page you saw exactly (keep its
     query string), only replacing the year by {{year}}. When several pages show the same
     listing (default view, bigger page size, "show all"), use the one that lists the most
     documents.
   - container: CSS selector of the element(s) that enclose the listing (every match is
     used); it must leave out menus, banners and footers. item: optional CSS selector of
     one entry (card, row) inside the container, whose text describes the document.
     title: optional CSS selector of the entry title inside the item.
     pagination_next_selector: CSS selector of the "next page" link when the listing is
     paginated (prefer a[rel=next]), else null. table_context: true when documents sit in
     table cells whose row or column headers tell their type or period.
   - CSS is lxml cssselect (CSS3: tag, #id, .class, [attr], [attr*=value], descendant and
     child combinators, :not(), :nth-child()). No :has(), no :contains(). Prefer stable ids
     and classes; avoid year-specific ids or generated class names.
   - default_language: ISO 639-1 code when every document is in one language, else null.
Keep reasoning under 60 words. Fields that do not apply to the chosen action are empty
lists or null."""


@dataclass
class VisitedPage:
    id: str
    url: str
    final_url: str
    title: str
    groups: list[LinkGroup]
    nav: list[NavLink]

    def group_ids(self) -> list[str]:
        return [f"{self.id}.G{i + 1}" for i in range(len(self.groups))]

    def nav_ids(self) -> list[str]:
        return [f"{self.id}.N{i + 1}" for i in range(len(self.nav))]


def digest_page(page: VisitedPage, base_url: str) -> str:
    lines = [f"== Page {page.id}: {page.final_url}", f"Title: {page.title}"]
    order = sorted(range(len(page.groups)), key=lambda i: -len(page.groups[i].candidates))
    shown = order[:MAX_GROUPS_PER_PAGE]
    lines.append(f"Document link groups ({len(page.groups)}):")
    for rank, index in enumerate(shown):
        group = page.groups[index]
        exts = ", ".join(f"{e or '?'} {n}" for e, n in group.extensions.most_common())
        lines.append(
            f"- {page.id}.G{index + 1}: {len(group.candidates)} links ({exts}) on "
            f"{', '.join(sorted(group.hosts))}; DOM path: {group.example_path}"
        )
        for cand in group.candidates[:3]:
            heading = f' [heading: "{cand.heading[:60]}"]' if cand.heading else ""
            tail = cand.url.split("?", 1)[0].rsplit("/", 1)[-1][:80]
            lines.append(f'  e.g. "{cand.link_text[:70] or cand.title[:70]}"{heading} -> {tail}')
        if rank < MAX_SNIPPETS_PER_PAGE and group.elements:
            entry = entry_element(group.elements[0], base_url)
            parent = entry.getparent()
            snippet = pruned_html(parent if parent is not None else entry, 900)
            lines.append(f"  HTML around one entry: {snippet}")
    if len(page.groups) > len(shown):
        lines.append(f"  ({len(page.groups) - len(shown)} smaller groups not shown)")
    lines.append(f"Navigation links ({len(page.nav)} shown):")
    for nav_id, nav in zip(page.nav_ids(), page.nav, strict=True):
        lines.append(f'- {nav_id}: "{nav.text}" -> {nav.url}')
    return "\n".join(lines)


@dataclass
class StructureResult:
    recipe: Recipe
    run: RunResult
    reply: StructureReply
    pages: dict[str, VisitedPage]
    expected: set[str]
    missing: list[str]
    extra: list[str]
    attempts: int

    @property
    def clean(self) -> bool:
        return self.run.status == "ok" and bool(self.run.documents) and not self.missing


class LearnError(Exception):
    pass


@dataclass
class LearnContext:
    issuer: str
    start_url: str
    years: Collection[int] | None
    today: date

    @property
    def recipe_name(self) -> str:
        return f"{self.issuer}-ir"

    @property
    def domain(self) -> str:
        return site_domain(host_of(self.start_url))


class Learner:
    def __init__(
        self,
        fetcher: Fetcher,
        llm: StructuredLLM,
        context: LearnContext,
        *,
        max_open_rounds: int = 2,
        max_repairs: int = 2,
        max_rule_repairs: int = 1,
        sample_size: int = 80,
    ) -> None:
        self.fetcher = fetcher
        self.llm = llm
        self.ctx = context
        self.max_open_rounds = max_open_rounds
        self.max_repairs = max_repairs
        self.max_rule_repairs = max_rule_repairs
        self.sample_size = sample_size
        self.pages: dict[str, VisitedPage] = {}
        self.notes: list[str] = []

    # -- pages -------------------------------------------------------------------------

    def visit(self, url: str) -> VisitedPage | None:
        if not in_site(host_of(url), self.ctx.domain):
            self.notes.append(f"not opened (other site): {url}")
            return None
        if any(normalize_url(p.url) == normalize_url(url) for p in self.pages.values()):
            return None
        try:
            page = self.fetcher.get(url, allowed_domains=[host_of(url)])
        except BlockedError:
            raise
        except FetchError as exc:
            self.notes.append(f"could not open {url}: {exc}")
            return None
        root = parse_html(page.content)
        page_id = f"P{len(self.pages) + 1}"
        visited = VisitedPage(
            id=page_id,
            url=url,
            final_url=page.final_url,
            title=page_title(root),
            groups=group_document_links(root, page.final_url),
            nav=navigation_links(root, page.final_url, self.ctx.domain, limit=MAX_NAV_PER_PAGE),
        )
        self.pages[page_id] = visited
        return visited

    def _nav_url(self, link_id: str) -> str | None:
        page_id, _, rest = link_id.partition(".")
        page = self.pages.get(page_id)
        if page is None or not rest.startswith("N") or not rest[1:].isdigit():
            return None
        index = int(rest[1:]) - 1
        return page.nav[index].url if 0 <= index < len(page.nav) else None

    def _group(self, group_id: str) -> LinkGroup | None:
        page_id, _, rest = group_id.partition(".")
        page = self.pages.get(page_id)
        if page is None or not rest.startswith("G") or not rest[1:].isdigit():
            return None
        index = int(rest[1:]) - 1
        return page.groups[index] if 0 <= index < len(page.groups) else None

    # -- structure ---------------------------------------------------------------------

    def _structure_messages(self, history: Sequence[Message]) -> list[Message]:
        years = ", ".join(str(y) for y in sorted(self.ctx.years)) if self.ctx.years else "all"
        digests = "\n\n".join(digest_page(p, p.final_url) for p in self.pages.values())
        notes = ("\nEngine notes: " + "; ".join(self.notes[-10:])) if self.notes else ""
        user = (
            f"Start page: {self.ctx.start_url}\nToday: {self.ctx.today.isoformat()}\n"
            f"Fiscal years checked now: {years} (the recipe must still cover every year)."
            f"{notes}\n\n{untrusted('pages', digests)}"
        )
        return [
            {"role": "system", "content": STRUCTURE_SYSTEM},
            {"role": "user", "content": user},
            *history,
        ]

    def learn_structure(self) -> StructureResult:
        if self.visit(self.ctx.start_url) is None:
            raise LearnError(f"start page unavailable: {'; '.join(self.notes)}")
        history: list[Message] = []
        best: StructureResult | None = None
        opened = repairs = 0
        for attempt in range(1, 2 + self.max_open_rounds + self.max_repairs):
            reply = self.llm.structured(
                self._structure_messages(history), StructureReply, step="structure"
            )
            history.append({"role": "assistant", "content": reply.model_dump_json()})
            if reply.action == "open_pages":
                if opened >= self.max_open_rounds:
                    history.append(_user("No more pages can be opened: return action recipe."))
                    continue
                opened += 1
                new = self._open(reply.open_links)
                text = f"Opened {new} new page(s); the digest above now includes them."
                history.clear()  # the digest is rebuilt with the new pages
                self.notes.append(text)
                continue
            try:
                recipe = self.build_recipe(reply)
            except (ValueError, ValidationError) as exc:
                diagnostic = f"The engine could not build a recipe from your reply: {exc}"
                log.info("structure attempt %d rejected: %s", attempt, exc)
            else:
                result = self._check_structure(recipe, reply, attempt)
                if best is None or _structure_score(result) > _structure_score(best):
                    best = result
                if result.clean and not result.extra:
                    break
                diagnostic = self._diagnostic(result)
            if repairs >= self.max_repairs:
                break
            repairs += 1
            history.append(_user(diagnostic))
        if best is None:
            raise LearnError("the LLM produced no usable recipe structure")
        return best

    def _open(self, link_ids: Sequence[str]) -> int:
        count = 0
        for link_id in list(link_ids)[:MAX_OPEN_PER_ROUND]:
            url = self._nav_url(link_id)
            if url is None:
                self.notes.append(f"unknown link id {link_id}")
                continue
            if self.visit(url) is not None:
                count += 1
        return count

    def build_recipe(
        self,
        reply: StructureReply,
        *,
        rules: ClassifyRules | None = None,
        health: HealthSpec | None = None,
        version: int = 1,
    ) -> Recipe:
        if not reply.sources:
            raise ValueError("no source given")
        chosen = [g for gid in reply.document_groups if (g := self._group(gid)) is not None]
        doc_hosts = {h for g in chosen for h in g.hosts}
        exts = {e for g in chosen for e in g.extensions if e in DOCUMENT_EXTENSIONS}
        exts |= {"pdf", "xlsx", "xls", "xlsb"}
        site = host_of(self.ctx.start_url)
        sources = []
        source_hosts = set()
        for i, draft in enumerate(reply.sources):
            url = draft.url.strip()
            placeholders = set(re.findall(r"\{([^}]*)\}", url))
            if placeholders - {"year"}:
                raise ValueError(f"source {url}: only the {{year}} placeholder is allowed")
            host = host_of(url.replace(_YEAR_PLACEHOLDER, "2000"))
            if not in_site(host, self.ctx.domain):
                raise ValueError(f"source {url} is not on the site {self.ctx.domain}")
            source_hosts.add(host)
            params = {}
            if _YEAR_PLACEHOLDER in url:
                first = draft.year_from or min(self.ctx.years or [self.ctx.today.year - 5])
                params["year"] = YearRange.model_validate({"from": first, "to": "current"})
            pagination = (
                NextLinkPagination(selector=draft.pagination_next_selector, max_pages=50)
                if draft.pagination_next_selector
                else None
            )
            sources.append(
                UrlTemplateSource(
                    id=f"listing-{i + 1}",
                    url=url,
                    params=params,
                    pagination=pagination,
                    items=ItemsSpec(
                        container=draft.container or None,
                        item=draft.item or None,
                        title=draft.title or None,
                        extensions=sorted(exts),
                        table_context=draft.table_context,
                    ),
                )
            )
        language = (reply.default_language or "").lower() or None
        if language and not re.fullmatch(r"[a-z]{2,3}", language):
            language = None
        return Recipe(
            schema_version=SCHEMA_VERSION,
            recipe=self.ctx.recipe_name,
            version=version,
            issuer=self.ctx.issuer,
            site=site,
            allowed_domains=sorted({site} | source_hosts | doc_hosts),
            defaults=Defaults(language=language),
            sources=sources,
            classify=rules or ClassifyRules(),
            health=health or HealthSpec(),
            provenance=Provenance(
                method="learned", created_at=self.ctx.today, notes=reply.reasoning[:300]
            ),
        )

    def check_years(self, recipe: Recipe) -> set[int] | None:
        """Years to replay while learning: the requested ones plus those of the listing
        pages already seen (so the check covers what the LLM looked at)."""
        if self.ctx.years is None:
            return None
        years = set(self.ctx.years)
        seen = {normalize_url(p.url) for p in self.pages.values()}
        seen |= {normalize_url(p.final_url) for p in self.pages.values()}
        for source in recipe.sources:
            for params, url in source.expand(self.ctx.today):
                if normalize_url(url) in seen:
                    years.update(params.values())
        return years

    def _check_structure(
        self, recipe: Recipe, reply: StructureReply, attempt: int
    ) -> StructureResult:
        run = run_recipe(recipe, self.fetcher, years=self.check_years(recipe), today=self.ctx.today)
        chosen = {gid for gid in reply.document_groups if self._group(gid) is not None}
        expected: set[str] = set()
        others: set[str] = set()
        for page in self.pages.values():
            for gid, group in zip(page.group_ids(), page.groups, strict=True):
                urls = {c.normalized_url for c in group.candidates}
                (expected if gid in chosen else others).update(urls)
        others -= expected
        found = {d.normalized_url for d in run.documents}
        result = StructureResult(
            recipe=recipe,
            run=run,
            reply=reply,
            pages=dict(self.pages),
            expected=expected,
            missing=sorted(expected - found),
            extra=sorted(found & others),
            attempts=attempt,
        )
        log.info(
            "structure attempt %d: %s, %d documents, %d missing, %d extra",
            attempt,
            run.status,
            len(found),
            len(result.missing),
            len(result.extra),
        )
        return result

    def _diagnostic(self, result: StructureResult) -> str:
        run = result.run
        lines = [
            "The engine ran your proposal.",
            f"- status: {run.status}" + (f" ({'; '.join(run.errors[:3])})" if run.errors else ""),
            f"- {len(run.documents)} documents found on {len(run.pages)} listing page(s):",
        ]
        lines += [f"  {v.url} -> {v.items} links" for v in run.pages[:8]]
        if result.missing:
            lines.append(
                f"- {len(result.missing)} documents of the groups you selected were NOT "
                "found, e.g.:"
            )
            lines += [f"  {u}" for u in result.missing[:8]]
        if result.extra:
            lines.append(
                f"- {len(result.extra)} documents of groups you did NOT select were found "
                "(menus, banners?), e.g.:"
            )
            lines += [f"  {u}" for u in result.extra[:5]]
        if not run.documents:
            lines.append("- No document at all: check the source URLs and the container.")
        lines.append("Fix the proposal and return action recipe.")
        return "\n".join(lines)

    # -- rules -------------------------------------------------------------------------

    def sample(self, docs: Sequence[FoundDocument], size: int | None = None) -> list[FoundDocument]:
        size = size or self.sample_size
        if len(docs) <= size:
            return list(docs)
        years = {str(y) for y in self.ctx.years or ()}

        def relevant(d: FoundDocument) -> bool:
            blob = d.url + " " + " ".join(v or "" for v in d.fields.values())
            return any(y in blob for y in years) or d.params.get("year") in (self.ctx.years or ())

        first = [d for d in docs if years and relevant(d)][:size]
        rest = [d for d in docs if d not in first]
        return first + rest[: size - len(first)]

    def _rules_prompt(
        self, structure: StructureResult, docs: Sequence[FoundDocument]
    ) -> tuple[list[Message], dict[str, FoundDocument]]:
        table, ids = document_table(docs)
        sources = "\n".join(f"- {s.url}" for s in structure.recipe.sources)
        user = (
            f"Issuer: {self.ctx.issuer}\nListing sources:\n{sources}\n"
            f"Default language of the site: {structure.recipe.defaults.language or 'none'}\n"
            f"{len(docs)} documents (of {len(structure.run.documents)} listed):\n"
            + untrusted("documents", table)
        )
        messages: list[Message] = [
            {"role": "system", "content": RULES_SYSTEM + taxonomy_text()},
            {"role": "user", "content": user},
        ]
        return messages, dict(zip(ids, docs, strict=True))

    def learn_rules(self, structure: StructureResult) -> RulesResult:
        docs = self.sample(structure.run.documents)
        messages, by_id = self._rules_prompt(structure, docs)
        best: RulesResult | None = None
        attempt = 0
        shrunk = False
        while attempt < 1 + self.max_rule_repairs:
            attempt += 1
            try:
                reply = self.llm.structured(messages, RulesReply, step="rules")
            except StructuredOutputError as exc:
                if best is not None:
                    log.info("rules repair failed (%s): keeping the best rules", exc)
                    break
                if shrunk or len(docs) <= MIN_RULES_SAMPLE:
                    raise
                # Usually a truncated reply: ask again about fewer documents.
                shrunk = True
                attempt -= 1
                docs = self.sample(docs, max(MIN_RULES_SAMPLE, len(docs) // 2))
                messages, by_id = self._rules_prompt(structure, docs)
                continue
            rules, errors = rules_from_reply(reply)
            evaluation = evaluate_rules(
                rules, reply.labels, by_id, structure.recipe.defaults.language
            )
            result = RulesResult(rules, reply, evaluation, errors, attempt)
            if best is None or result.score > best.score:
                best = result
            log.info(
                "rules attempt %d: type %.0f%%, period %.0f%%, %d invalid rules",
                attempt,
                100 * evaluation.type_agreement,
                100 * evaluation.period_agreement,
                len(errors),
            )
            if result.good:
                break
            messages = [
                *messages,
                {"role": "assistant", "content": reply.model_dump_json()},
                _user(rules_diagnostic(result)),
            ]
        if best is None:  # pragma: no cover - the loop runs at least once
            raise LearnError("no rules")
        best.decisions = {
            by_id[label.id].normalized_url: label.decision("llm:learn")
            for label in best.reply.labels
            if label.id in by_id
        }
        return best


def _user(text: str) -> Message:
    return {"role": "user", "content": text}


def _structure_score(result: StructureResult) -> tuple[int, int, int, int]:
    return (
        int(result.run.status == "ok"),
        -len(result.missing),
        -len(result.extra),
        len(result.run.documents),
    )


# -- rules step ----------------------------------------------------------------------

RuleTarget = Literal[
    "url",
    "link_text",
    "title",
    "heading",
    "row_header",
    "column_header",
    "link_title",
    "context",
    "any",
]


class TypeRuleDraft(_Reply):
    pattern: str
    target: RuleTarget
    type: DocumentType


class PeriodRuleDraft(_Reply):
    pattern: str
    target: RuleTarget


class LanguageRuleDraft(_Reply):
    pattern: str
    target: RuleTarget
    language: str | None


class RulesReply(_Reply):
    reasoning: str
    type_rules: list[TypeRuleDraft]
    period_rules: list[PeriodRuleDraft]
    language_rules: list[LanguageRuleDraft]
    labels: list[DocumentLabel]


RULES_SYSTEM = f"""You write the classification rules of a retrievr recipe, and you label
documents. Rules are Python regular expressions, matched case-insensitively. The engine
applies them to every document of the site, now and in future years, so they must rely
on stable patterns of this site (file names, link labels, headings, table headers),
never on one year or one URL. {UNTRUSTED_NOTICE}

Each rule has a pattern and a target, the document field it is searched in: url (decoded
URL path, without the query string), link_text, title, heading (nearest heading above the
link), row_header and column_header (table cells), link_title (title attribute), context
(text of the entry), or any (url, then title and context).
- type_rules: ordered, the first match wins, so put specific rules before general ones
  (e.g. "fixed income ... transcript" before "transcript"). Each gives one type id. Cover
  every kind of document you see; a document that no rule matches stays undecided.
- period_rules: ordered, the first match wins. Named groups: year (4 or 2 digits,
  required) and optionally q (the quarter: "Q3", "3Q" or "T3" all work, it must contain
  the digit), h (the half: "H1", "S1", "1H") or fy (any text meaning full year). Without
  q, h or fy the period is the year only. Example: (?P<q>Q[1-4])[ _-]?(?P<year>20\\d{{2}}).
- language_rules: either a named group lang (2-letter code) or a fixed language value.
  Leave empty when the site default language is right for every document.
Then label every document id with its type, period and language: your own judgement,
which the engine compares with what the rules give.
Keep reasoning under 60 words.

"""


@dataclass
class RulesEvaluation:
    type_agreement: float
    period_agreement: float
    disagreements: list[str]


@dataclass
class RulesResult:
    rules: ClassifyRules
    reply: RulesReply
    evaluation: RulesEvaluation
    errors: list[str]
    attempts: int
    decisions: dict[str, Decision] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return (
            self.evaluation.type_agreement
            + self.evaluation.period_agreement
            - 0.01 * len(self.errors)
        )

    @property
    def good(self) -> bool:
        ev = self.evaluation
        return (
            not self.errors
            and ev.type_agreement >= RULES_AGREEMENT
            and ev.period_agreement >= RULES_AGREEMENT
        )


def rules_from_reply(reply: RulesReply) -> tuple[ClassifyRules, list[str]]:
    """Valid rules of the reply, and why the others were dropped."""
    errors: list[str] = []

    def keep(model: type[M], raw: Iterable[BaseModel]) -> list[M]:
        kept = []
        for draft in raw:
            try:
                kept.append(model.model_validate(draft.model_dump()))
            except ValidationError as exc:
                message = exc.errors()[0]["msg"]
                errors.append(f"{model.__name__} {draft.model_dump_json()}: {message}")
        return kept

    rules = ClassifyRules(
        type=keep(TypeRule, reply.type_rules),
        period=keep(PeriodRule, reply.period_rules),
        language=[
            r
            for r in keep(LanguageRule, reply.language_rules)
            if r.language is None or re.fullmatch(r"[a-z]{2,3}", r.language)
        ],
    )
    return rules, errors


def evaluate_rules(
    rules: ClassifyRules,
    labels: Sequence[DocumentLabel],
    docs: Mapping[str, FoundDocument],
    default_language: str | None,
) -> RulesEvaluation:
    type_ok = period_ok = period_total = 0
    labelled = 0
    disagreements: list[str] = []
    for label in labels:
        doc = docs.get(label.id)
        if doc is None:
            continue
        labelled += 1
        cls = classify(rules, url=doc.url, fields=doc.fields, default_language=default_language)
        good_type = cls.type == label.type
        type_ok += good_type
        good_period = True
        if label.period is not None:
            period_total += 1
            good_period = cls.period == label.period
            period_ok += good_period
        if not (good_type and good_period):
            disagreements.append(
                f"{label.id}: rules give type={cls.type} period={cls.period}, your label "
                f"type={label.type} period={label.period}"
            )
    return RulesEvaluation(
        type_agreement=type_ok / labelled if labelled else 0.0,
        period_agreement=period_ok / period_total if period_total else 1.0,
        disagreements=disagreements,
    )


def rules_diagnostic(result: RulesResult) -> str:
    ev = result.evaluation
    lines = [
        f"The engine applied your rules to the documents: type agrees with your labels "
        f"for {ev.type_agreement:.0%}, period for {ev.period_agreement:.0%}."
    ]
    if result.errors:
        lines.append("Invalid rules (dropped):")
        lines += [f"- {e}" for e in result.errors[:10]]
    if ev.disagreements:
        lines.append(f"Disagreements ({len(ev.disagreements)}), e.g.:")
        lines += [f"- {d}" for d in ev.disagreements[:30]]
    lines.append("Return corrected rules (and labels if they were wrong).")
    return "\n".join(lines)


# -- health --------------------------------------------------------------------------


def health_for(docs: Sequence[FoundDocument]) -> HealthSpec:
    """Health expectations stored with a learned recipe (checked by later runs)."""
    per_year = Counter(d.fiscal_year for d in docs if d.fiscal_year is not None)
    items = None
    if per_year:
        recent = [n for y, n in sorted(per_year.items())[-3:]]
        low = max(1, math.floor(min(recent) * 0.5))
        items = (low, max(low + 1, math.ceil(max(recent) * 2)))
    periods = {d.period or "" for d in docs}
    if any("-Q" in p for p in periods):
        cadence = Cadence(every="quarter", max_gap_days=110)
    elif any("-H" in p for p in periods):
        cadence = Cadence(every="half_year", max_gap_days=200)
    else:
        cadence = Cadence(every="year", max_gap_days=400)
    typed = [d for d in docs if d.type is not None and d.fiscal_year is not None]
    anchors: list[str] = []
    if typed:
        latest = max(d.fiscal_year or 0 for d in typed)
        seen_types: set[str] = set()
        for doc in (d for d in typed if d.fiscal_year == latest):
            if doc.type not in seen_types:
                anchors.append(doc.url)
                seen_types.add(doc.type or "")
            if len(anchors) == 3:
                break
    return HealthSpec(anchors=anchors, items_per_year=items, cadence=cadence)


def classify_remaining(
    llm: StructuredLLM,
    docs: Sequence[FoundDocument],
    decisions: Mapping[str, Decision],
    *,
    issuer: str,
    years: Collection[int] | None,
) -> dict[str, Decision]:
    """LLM decisions for documents the rules and earlier decisions leave undecided.

    Only documents of the requested years (or of no known year) are sent."""
    pending = [
        d
        for d in apply_decisions(docs, decisions)
        if needs_decision(d)
        and d.normalized_url not in decisions
        and (years is None or d.fiscal_year is None or d.fiscal_year in years)
    ]
    if not pending:
        return {}
    return classify_with_llm(llm, pending, issuer=issuer)


# -- request -> filter ---------------------------------------------------------------


class Filter(_Reply):
    """A compiled request: applied to documents without any LLM call."""

    types: list[DocumentType] | None = None
    """Types to keep; null keeps every type."""
    exclude_types: list[DocumentType] = Field(default_factory=list)
    year_from: int | None = None
    """First fiscal year (inclusive); null for no lower bound."""
    year_to: int | None = None
    """Last fiscal year (inclusive); null for no upper bound."""
    languages: list[str] | None = None
    """ISO 639-1 codes to keep; null keeps every language."""

    @field_validator("languages")
    @classmethod
    def _lower(cls, value: list[str] | None) -> list[str] | None:
        return [v.lower() for v in value] if value else None

    def matches(self, type_: str | None, fiscal_year: int | None, language: str | None) -> bool:
        if self.types is not None and type_ not in self.types:
            return False
        if type_ is not None and type_ in self.exclude_types:
            return False
        if self.year_from is not None or self.year_to is not None:
            if fiscal_year is None:
                return False
            if self.year_from is not None and fiscal_year < self.year_from:
                return False
            if self.year_to is not None and fiscal_year > self.year_to:
                return False
        return self.languages is None or language in self.languages

    def select(self, docs: Iterable[FoundDocument]) -> list[FoundDocument]:
        return [d for d in docs if self.matches(d.type, d.fiscal_year, d.language)]


FILTER_SYSTEM = """You compile a request for documents into a JSON filter. The filter is
applied by code to documents already typed with the ids below.
- types: every type id the request asks for (null when it asks for all documents or names
  no kind of document). Map words to ids generously but precisely: "results documents",
  "all documents" or "all the annual documents of 2025" is null (the year is the only
  constraint); "annual financial reports" (rapports financiers annuels) is annual_report
  and urd, since a registration document contains the annual financial report; "annual
  reports" alone is annual_report; "registration documents" is
  urd; "Form 20-F" or "10-K" is annual_report; "10-Q" is quarterly_report; "data packs" or
  "financial supplements" is financial_data_supplement; "transcripts" without more detail
  is analyst_call_transcript (add fixed_income_call_transcript when fixed income calls
  are named).
- exclude_types: types the request explicitly excludes ("no press releases").
- year_from, year_to: fiscal years, inclusive. "since 2024" is year_from 2024 and year_to
  null; "for 2024" or "2024 only" is 2024 to 2024; "2024 and 2025" is 2024 to 2025.
  Relative phrases use today's date.
- languages: only when the request restricts the language ("English only" is ["en"]).
- File formats (PDF, Excel, XBRL) and issuer or entity names are not part of the filter.
"""


def compile_request(llm: StructuredLLM, request: str, *, today: date) -> Filter:
    messages: list[Message] = [
        {"role": "system", "content": FILTER_SYSTEM + taxonomy_text()},
        {"role": "user", "content": f"Today: {today.isoformat()}\nRequest: {request}"},
    ]
    return llm.structured(messages, Filter, step="filter")


def filter_json(f: Filter) -> str:
    return json.dumps(f.model_dump(mode="json"), sort_keys=True)


def is_inside(recipe: Recipe, docs: Iterable[FoundDocument]) -> bool:
    return all(is_allowed(d.url, recipe.allowed_domains) for d in docs)
