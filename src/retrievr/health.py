"""Health checks: is the active recipe still describing the site? (deterministic, no LLM)

A replay of the recipe is compared with what earlier healthy runs established (the
documents they listed, with their metadata) and with the expectations stored in the
recipe (``health``: anchors, documents per year, publication cadence).

Verdicts, from the least to the most severe:

- ``ok``: nothing unexpected;
- ``stale``: the recipe works but nothing was published for longer than the cadence
  allows (a new listing page may exist that the recipe does not know);
- ``suspect``: something is off but the output may still be right (a few known documents
  missing, an unusual volume, the site unreachable): reported, not repaired;
- ``broken``: the recipe no longer lists what it listed (no document, an anchor or many
  known documents missing, metadata of known documents changed): repair is needed;
- ``blocked``: the site refused us; nothing to repair.

The checks are built so that a recipe that lost documents or misreads them never gets
``ok`` ("zero false ok"): any known document missing while still online is at least
``suspect``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from retrievr.recipe import FoundDocument, ParamValue, Recipe, RequestSource, RunResult
from retrievr.urls import normalize_url

Verdict = Literal["ok", "stale", "suspect", "broken", "blocked"]
SEVERITY: dict[Verdict, int] = {"ok": 0, "stale": 1, "suspect": 2, "broken": 3, "blocked": 4}

MISSING_BROKEN_RATIO = 0.05
"""More known documents missing than this (and still online): broken."""
DRIFT_BROKEN_RATIO = 0.10
"""More known documents whose type or period changed than this: broken."""
VOLUME_SUSPECT_RATIO = 1.5
PROBE_SAMPLE = 5
"""Missing documents checked online before deciding they were removed from the site."""
PUBLICATION_LAG_DAYS = 60
"""Estimated delay between the end of a period and its publication, when no date."""

Probe = Callable[[str], bool | None]
"""``url -> True`` (still online), ``False`` (gone: 404/410) or ``None`` (unknown)."""


@dataclass(frozen=True)
class KnownDocument:
    """A document an earlier healthy run listed (internal state)."""

    normalized_url: str
    url: str
    title: str
    source_id: str
    params: Mapping[str, ParamValue]
    type: str | None
    period: str | None
    fiscal_year: int | None
    language: str | None

    @property
    def listed_year(self) -> int | None:
        """The year of the listing page it was found on (else its fiscal year)."""
        for name in ("year", "publication_year"):
            value = self.params.get(name)
            if isinstance(value, int):
                return value
        return self.fiscal_year

    @classmethod
    def of(cls, doc: FoundDocument) -> KnownDocument:
        return cls(
            doc.normalized_url,
            doc.url,
            doc.title,
            doc.source_id,
            dict(doc.params),
            doc.type,
            doc.period,
            doc.fiscal_year,
            doc.language,
        )


@dataclass(frozen=True)
class HealthCheck:
    name: str
    verdict: Verdict
    """What this check alone implies (``ok`` when it passed)."""
    detail: str


@dataclass
class HealthReport:
    site: str
    recipe: str
    verdict: Verdict = "ok"
    checks: list[HealthCheck] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    """Known documents (normalized URLs) not listed any more and not known to be gone."""
    removed: list[str] = field(default_factory=list)
    """Known documents not listed any more because they are gone from the site."""
    drifted: list[str] = field(default_factory=list)
    """Known documents listed with another type or period than before."""
    baseline: bool = False
    """No history yet: only the checks that need none were made."""

    @property
    def repair_needed(self) -> bool:
        return self.verdict == "broken"

    def add(self, name: str, verdict: Verdict, detail: str) -> None:
        self.checks.append(HealthCheck(name, verdict, detail))
        if SEVERITY[verdict] > SEVERITY[self.verdict]:
            self.verdict = verdict

    def failures(self) -> list[HealthCheck]:
        return [c for c in self.checks if c.verdict != "ok"]

    def summary(self) -> str:
        bad = self.failures()
        if not bad:
            return f"{self.verdict}: {len(self.checks)} checks passed"
        return f"{self.verdict}: " + "; ".join(f"{c.name} {c.verdict} ({c.detail})" for c in bad)


def scope_years(recipe: Recipe, years: Collection[int] | None) -> set[int] | None:
    """Listing years a run restricted to ``years`` covers (None: every year).

    Request sources with year selectors also read the bucket of the next year."""
    if years is None:
        return None
    scope = set(years)
    if any(isinstance(s, RequestSource) and s.params for s in recipe.sources):
        scope |= {y + 1 for y in years}
    return scope


def in_scope(doc: KnownDocument, scope: set[int] | None) -> bool:
    year = doc.listed_year
    return scope is None or year is None or year in scope


def assess(
    recipe: Recipe,
    run: RunResult,
    documents: Sequence[FoundDocument],
    known: Sequence[KnownDocument],
    *,
    site: str,
    today: date,
    years: Collection[int] | None = None,
    probe: Probe | None = None,
) -> HealthReport:
    """Health of ``recipe`` from one replay (``documents``: the run's documents after
    stored decisions were applied) against the ``known`` documents of earlier runs."""
    report = HealthReport(site=site, recipe=recipe.key, baseline=not known)
    if run.status == "blocked":
        report.add("run", "blocked", "; ".join(run.errors[:2]))
        return report
    if run.status == "error":
        report.add("run", _error_verdict(run.errors), "; ".join(run.errors[:3]))
    else:
        report.add("run", "ok", f"{len(run.pages)} listing pages")
    if not documents:
        report.add("documents", "broken", "no document listed")
        return report
    report.add("documents", "ok", f"{len(documents)} listed")

    scope = scope_years(recipe, years)
    found = {d.normalized_url: d for d in documents}
    scoped = [k for k in known if in_scope(k, scope)]
    _anchors(report, recipe, found, scoped, known, years, probe)
    _known_documents(report, found, scoped, probe)
    _metadata(report, found, scoped)
    _volume(report, documents, scoped, known, scope)
    _items_per_year(report, recipe, documents, scope, today, known)
    _cadence(report, recipe, documents, scope, today)
    return report


def _error_verdict(errors: Sequence[str]) -> Verdict:
    """A listing URL that answers 404/410 or a response the recipe cannot read is a
    structural change; timeouts and server errors are reported, not repaired."""
    structural = ("HTTP 404", "HTTP 410", "JSON", "path", "unreadable", "expected")
    return "broken" if any(s in e for e in errors for s in structural) else "suspect"


def _probe_missing(missing: Sequence[str], probe: Probe | None) -> tuple[list[str], list[str]]:
    """Split missing documents into (still online or unknown, gone from the site)."""
    if probe is None or not missing:
        return list(missing), []
    sample = sorted(missing)[:PROBE_SAMPLE]
    states = {url: probe(url) for url in sample}
    gone = [u for u, s in states.items() if s is False]
    if len(gone) == len(sample) and len(missing) > len(sample):
        # Every sampled document is gone: the site removed them, the listing is fine.
        return [], sorted(missing)
    return [u for u in missing if u not in gone], gone


def _anchors(
    report: HealthReport,
    recipe: Recipe,
    found: Mapping[str, FoundDocument],
    scoped: Sequence[KnownDocument],
    known: Sequence[KnownDocument],
    years: Collection[int] | None,
    probe: Probe | None,
) -> None:
    anchors = [normalize_url(a) for a in recipe.health.anchors]
    if not anchors:
        return
    known_urls = {k.normalized_url for k in known}
    scoped_urls = {k.normalized_url for k in scoped}
    # An anchor outside the years of this run is not checked; one never seen yet is
    # checked on unrestricted runs only.
    checked = [a for a in anchors if a in scoped_urls or (a not in known_urls and years is None)]
    absent = [a for a in checked if a not in found]
    online, gone = _probe_missing(absent, probe)
    if online:
        report.add("anchors", "broken", f"{len(online)} of {len(checked)} missing: {online[0]}")
    else:
        note = f", {len(gone)} gone from the site" if gone else ""
        report.add("anchors", "ok", f"{len(checked)} checked{note}")


def _known_documents(
    report: HealthReport,
    found: Mapping[str, FoundDocument],
    scoped: Sequence[KnownDocument],
    probe: Probe | None,
) -> None:
    if not scoped:
        report.add("known_documents", "ok", "no history for these years (baseline)")
        return
    absent = [k.normalized_url for k in scoped if k.normalized_url not in found]
    report.missing, report.removed = _probe_missing(absent, probe)
    still = len(scoped) - len(report.removed)
    lost = len(report.missing)
    detail = f"{lost} of {still} missing" + (
        f", {len(report.removed)} gone from the site" if report.removed else ""
    )
    if lost and lost > MISSING_BROKEN_RATIO * still:
        report.add("known_documents", "broken", detail)
    elif lost or len(report.removed) > 0.2 * len(scoped):
        report.add("known_documents", "suspect", detail)
    else:
        report.add("known_documents", "ok", detail)


def _metadata(
    report: HealthReport, found: Mapping[str, FoundDocument], scoped: Sequence[KnownDocument]
) -> None:
    pairs = [(k, found[k.normalized_url]) for k in scoped if k.normalized_url in found]
    compared = [(k, d) for k, d in pairs if k.type is not None or k.period is not None]
    if not compared:
        return
    report.drifted = sorted(
        k.normalized_url for k, d in compared if (k.type, k.period) != (d.type, d.period)
    )
    changed = len(report.drifted)
    detail = f"{changed} of {len(compared)} known documents changed type or period"
    if changed > DRIFT_BROKEN_RATIO * len(compared):
        report.add("metadata", "broken", detail)
    elif changed:
        report.add("metadata", "suspect", detail)
    else:
        report.add("metadata", "ok", detail)


def _volume(
    report: HealthReport,
    documents: Sequence[FoundDocument],
    scoped: Sequence[KnownDocument],
    known: Sequence[KnownDocument],
    scope: set[int] | None,
) -> None:
    if len(scoped) < 5:
        return
    outside = {k.normalized_url for k in known} - {k.normalized_url for k in scoped}
    listed = [
        d
        for d in documents
        if d.normalized_url not in outside and in_scope(KnownDocument.of(d), scope)
    ]
    limit = VOLUME_SUSPECT_RATIO * len(scoped)
    verdict: Verdict = "suspect" if len(listed) > limit else "ok"
    report.add("volume", verdict, f"{len(listed)} listed vs {len(scoped)} known")


def _items_per_year(
    report: HealthReport,
    recipe: Recipe,
    documents: Sequence[FoundDocument],
    scope: set[int] | None,
    today: date,
    known: Sequence[KnownDocument] = (),
) -> None:
    bounds = recipe.health.items_per_year
    if bounds is None:
        return
    low, high = bounds
    counts = Counter(d.fiscal_year for d in documents if d.fiscal_year is not None)
    years = set(counts) if scope is None else set(scope)
    # Too few documents is a regression only for a year the site listed before: a site
    # that never listed a year (its page holds recent years only) has nothing to lose.
    listed_before = {k.fiscal_year for k in known if k.fiscal_year is not None}
    out = []
    for year in sorted(years):
        n = counts.get(year, 0)
        complete = year <= today.year - 2  # every document of the year is published
        if n > high or (complete and n < low and (not known or year in listed_before)):
            out.append(f"{year}: {n}")
    verdict: Verdict = "suspect" if out else "ok"
    detail = f"outside [{low}, {high}]: {', '.join(out)}" if out else f"within [{low}, {high}]"
    report.add("items_per_year", verdict, detail)


def _cadence(
    report: HealthReport,
    recipe: Recipe,
    documents: Sequence[FoundDocument],
    scope: set[int] | None,
    today: date,
) -> None:
    cadence = recipe.health.cadence
    if cadence is None or (scope is not None and today.year not in scope):
        return  # past years only: nothing new is expected
    latest = max((d for d in (published(doc) for doc in documents) if d), default=None)
    if latest is None:
        return
    gap = (today - latest).days
    verdict: Verdict = "stale" if gap > cadence.max_gap_days else "ok"
    report.add("cadence", verdict, f"latest publication {latest.isoformat()} ({gap} days ago)")


def published(doc: FoundDocument) -> date | None:
    """Publication date, or an estimate from the end of its period."""
    if doc.date is not None:
        return doc.date
    end = period_end(doc.period, doc.fiscal_year)
    return end + timedelta(days=PUBLICATION_LAG_DAYS) if end else None


def period_end(period: str | None, fiscal_year: int | None) -> date | None:
    if period is None:
        return date(fiscal_year, 12, 31) if fiscal_year else None
    year_text, _, part = period.partition("-")
    if not year_text.isdigit():
        return None
    year = int(year_text)
    ends = {"Q1": (3, 31), "Q2": (6, 30), "Q3": (9, 30), "Q4": (12, 31), "H1": (6, 30)}
    month, day = ends.get(part, (12, 31))
    return date(year, month, day)
