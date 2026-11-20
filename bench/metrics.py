"""Benchmark metrics: recall, precision, metadata accuracy, snapshot coverage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from bench.model import GroundTruth
from retrievr.candidates import parse_html, select
from retrievr.recipe import FoundDocument
from retrievr.snapshots import load_index
from retrievr.urls import normalize_url, resolve_url

METADATA_FIELDS = ("type", "period", "fiscal_year", "language")


@dataclass(frozen=True)
class DocumentScore:
    expected: int
    found: int
    """Documents returned in the ground-truth window (or without fiscal year)."""
    matched: int
    recall: float
    precision: float
    metadata_accuracy: dict[str, float]
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)


def score_documents(found: Iterable[FoundDocument], truth: GroundTruth) -> DocumentScore:
    """Compare a run with the ground truth.

    Recall counts every returned document. Precision only looks at documents inside the
    ground-truth window; a document without fiscal year counts as inside (it could be
    anything), so an unclassified run cannot look precise.
    """
    expected = truth.by_url()
    docs = {d.normalized_url: d for d in found}
    in_window = {
        u: d
        for u, d in docs.items()
        if d.fiscal_year is None or d.fiscal_year in truth.fiscal_years
    }
    matched = [u for u in expected if u in docs]
    accuracy = {
        name: (
            sum(getattr(docs[u], name) == getattr(expected[u], name) for u in matched)
            / len(matched)
            if matched
            else 0.0
        )
        for name in METADATA_FIELDS
    }
    correct_in_window = sum(1 for u in in_window if u in expected)
    return DocumentScore(
        expected=len(expected),
        found=len(in_window),
        matched=len(matched),
        recall=len(matched) / len(expected),
        precision=correct_in_window / len(in_window) if in_window else 0.0,
        metadata_accuracy=accuracy,
        missing=sorted(u for u in expected if u not in docs),
        extra=sorted(u for u in in_window if u not in expected),
    )


@dataclass(frozen=True)
class Coverage:
    ratio: float
    missing: list[str]


def snapshot_links(directory: Path) -> set[str]:
    """Normalized URLs of every link found in the HTML bodies of a snapshot."""
    links: set[str] = set()
    for entry in load_index(directory).responses:
        if entry.file is None or not entry.file.endswith(".html"):
            continue
        root = parse_html((directory / entry.file).read_bytes())
        for a in select(root, "a[href]"):
            href = a.get("href").strip()
            if href and not href.lower().startswith(("javascript:", "mailto:", "#")):
                links.add(normalize_url(resolve_url(href, entry.url)))
    return links


def snapshot_coverage(directory: Path, truth: GroundTruth) -> Coverage:
    """Share of ground-truth documents linked from the captured pages.

    100 % means the snapshot is enough to evaluate a recipe (or a learner) offline.
    """
    links = snapshot_links(directory)
    expected = truth.by_url()
    missing = sorted(u for u in expected if u not in links)
    return Coverage(ratio=1 - len(missing) / len(expected), missing=missing)
