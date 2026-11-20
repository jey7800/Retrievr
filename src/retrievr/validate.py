"""Validation gates a candidate recipe must pass before it is activated (rule 5).

Deterministic, never calls an LLM. Gates that compare with history (anchors, known
documents, volume) apply when a previous recipe exists (repair); a recipe learned on a
new site only has the others.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

from retrievr.recipe import Recipe, RunResult
from retrievr.urls import is_allowed

KNOWN_DOCUMENTS_RATIO = 0.95
MAX_VOLUME_RATIO = 1.5


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class History:
    """What the previous recipe of a site established."""

    anchors: Collection[str]
    """Normalized URLs that must all be found."""
    known_online: Collection[str]
    """Normalized URLs of previously found documents that are still online."""
    volume: int
    """Documents the previous recipe returned on its last healthy run."""


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        return "; ".join(
            f"{c.name}: {'ok' if c.passed else 'FAIL'} ({c.detail})" for c in self.checks
        )


def signature(run: RunResult) -> list[tuple[str, str | None, str | None, str | None]]:
    """What two runs must agree on: documents and their metadata."""
    return sorted((d.normalized_url, d.type, d.period, d.language) for d in run.documents)


def validate_runs(
    recipe: Recipe,
    first: RunResult,
    second: RunResult | None,
    *,
    history: History | None = None,
) -> ValidationReport:
    report = ValidationReport()
    add = report.checks.append
    add(Check("runs", first.status == "ok", first.status + _errors(first)))
    if second is not None:
        add(Check("second_run", second.status == "ok", second.status + _errors(second)))
    count = len(first.documents)
    add(Check("documents", count > 0, f"{count} found"))
    outside = [d.url for d in first.documents if not is_allowed(d.url, recipe.allowed_domains)]
    add(Check("domains", not outside, f"{len(outside)} outside allowed_domains"))
    if second is not None:
        same = signature(first) == signature(second)
        add(Check("deterministic", same, "identical" if same else "the two runs differ"))
    if history is not None:
        found = {d.normalized_url for d in first.documents}
        missing_anchors = [a for a in history.anchors if a not in found]
        add(Check("anchors", not missing_anchors, f"{len(missing_anchors)} missing"))
        known = set(history.known_online)
        ratio = len(known & found) / len(known) if known else 1.0
        add(
            Check(
                "known_documents",
                ratio >= KNOWN_DOCUMENTS_RATIO,
                f"{ratio:.0%} of {len(known)} still found",
            )
        )
        limit = MAX_VOLUME_RATIO * max(history.volume, 1)
        add(Check("volume", count <= limit, f"{count} vs at most {limit:.0f}"))
    return report


def _errors(run: RunResult) -> str:
    return f": {'; '.join(run.errors[:3])}" if run.errors else ""
