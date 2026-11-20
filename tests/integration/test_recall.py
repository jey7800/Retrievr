"""Milestone 1 acceptance: 100 % recall against the benchmark ground truth, from snapshots.

One parametrized suite for every hand-written recipe. Ground truth covers a window of
fiscal years; documents outside that window are ignored, but every returned document
must have a fiscal year so that nothing can escape the precision check.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from bench.model import Document, load_ground_truth, load_sites

from retrievr.download import Downloader, Manifest
from retrievr.recipe import FoundDocument, Recipe, RunResult, load_recipe, run_recipe
from retrievr.snapshots import ReplayTransport
from tests.conftest import RECIPES, serve_site

TODAY = date(2026, 11, 20)
COMPARED_FIELDS = ("type", "period", "fiscal_year", "language", "date", "title")


@dataclass(frozen=True)
class Site:
    fixture: str
    recipe: str
    expected: int
    """Number of ground-truth documents (guards against an emptied ground truth)."""

    def __str__(self) -> str:
        return self.fixture


SITES = [
    Site("societegenerale", "societegenerale-ir@1.yaml", 30),
    Site("deutschebank", "deutschebank-ir@1.yaml", 62),
    Site("ubs", "ubs-ir@1.yaml", 39),
]


@dataclass
class Case:
    site: Site
    snapshot: ReplayTransport
    recipe: Recipe
    truth: dict[str, Document]
    years: list[int]

    def run(self, make_fetcher: Any) -> RunResult:
        return run_recipe(self.recipe, make_fetcher(), years=self.years, today=TODAY)

    def in_window(self, result: RunResult) -> list[FoundDocument]:
        return [d for d in result.documents if d.fiscal_year in self.years]


@pytest.fixture(params=SITES, ids=str)
def case(request: pytest.FixtureRequest) -> Iterator[Case]:
    site: Site = request.param
    bench_site = next(s for s in load_sites().sites if s.id == site.fixture)
    assert bench_site.recipe == site.recipe and bench_site.ground_truth
    gt = load_ground_truth(bench_site.ground_truth_path())
    truth = gt.by_url()
    assert len(truth) == site.expected
    with serve_site(site.fixture) as snapshot:
        yield Case(site, snapshot, load_recipe(RECIPES / site.recipe), truth, gt.fiscal_years)


def test_recall_is_100_percent(case: Case, make_fetcher: Any) -> None:
    result = case.run(make_fetcher)
    assert result.status == "ok", result.errors
    found = {d.normalized_url for d in result.documents}
    missing = sorted(set(case.truth) - found)
    assert not missing, f"recall {1 - len(missing) / len(case.truth):.0%}, missing: {missing}"


def test_no_extra_documents_in_window(case: Case, make_fetcher: Any) -> None:
    result = case.run(make_fetcher)
    undated = [d.url for d in result.documents if d.fiscal_year is None]
    assert not undated, "documents without fiscal year escape the precision check"
    extra = sorted({d.normalized_url for d in case.in_window(result)} - set(case.truth))
    assert not extra
    assert not case.snapshot.misses


def test_metadata_matches_ground_truth(case: Case, make_fetcher: Any) -> None:
    mismatches = []
    for doc in case.in_window(case.run(make_fetcher)):
        expected = case.truth[doc.normalized_url]
        keys = [k for k in COMPARED_FIELDS if getattr(expected, k) is not None]
        actual = {k: getattr(doc, k) for k in keys}
        wanted = {k: getattr(expected, k) for k in keys}
        if actual != wanted:
            mismatches.append((doc.url, actual, wanted))
    assert not mismatches


def test_every_document_is_typed(case: Case, make_fetcher: Any) -> None:
    untyped = [d.url for d in case.in_window(case.run(make_fetcher)) if d.type is None]
    assert not untyped


def test_two_runs_give_identical_results(case: Case, make_fetcher: Any) -> None:
    assert case.run(make_fetcher).documents == case.run(make_fetcher).documents


def test_download_tree_manifest_and_dedup(case: Case, make_fetcher: Any, tmp_path: Path) -> None:
    fetcher = make_fetcher()
    documents = case.in_window(case.run(make_fetcher))
    outcomes = Downloader(fetcher, tmp_path).download_all(case.recipe, documents)
    failed = [(o.document.url, o.status, o.reason) for o in outcomes if o.status != "downloaded"]
    assert not failed

    manifest = Manifest.load(tmp_path)
    paths = [e.path for e in manifest.entries]
    assert len(paths) == len(set(paths)) == case.site.expected
    for entry in manifest.entries:
        assert (tmp_path / entry.path).is_file()
        assert entry.path.startswith(f"{case.recipe.issuer}/{entry.fiscal_year}/")
        assert (entry.recipe, entry.recipe_version) == (case.recipe.recipe, case.recipe.version)

    again = Downloader(fetcher, tmp_path).download_all(case.recipe, documents)
    assert {o.status for o in again} == {"known_url"}
