"""Milestone 1 acceptance: 100 % recall on the hand-made ground truth, from snapshots."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from retrievr.download import Downloader, Manifest
from retrievr.recipe import RunResult, load_recipe, run_recipe
from retrievr.urls import normalize_url
from tests.conftest import FIXTURES, RECIPES, SiteSnapshot

TODAY = date(2026, 11, 20)
GROUND_TRUTH = FIXTURES / "societegenerale" / "ground_truth.yaml"
RECIPE = RECIPES / "societegenerale-ir@1.yaml"


def _ground_truth() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(GROUND_TRUTH.read_text("utf-8"))
    return {normalize_url(d["url"]): d for d in data["documents"]}


def _run(make_fetcher: Any) -> RunResult:
    recipe = load_recipe(RECIPE)
    return run_recipe(recipe, make_fetcher(), years=[2024, 2025], today=TODAY)


def test_recall_is_100_percent(sg_site: SiteSnapshot, make_fetcher: Any) -> None:
    result = _run(make_fetcher)
    truth = _ground_truth()
    found = {d.normalized_url for d in result.documents}

    assert result.status == "ok", result.errors
    missing = sorted(set(truth) - found)
    assert not missing, f"recall {1 - len(missing) / len(truth):.0%}, missing: {missing}"


def test_no_extra_documents(sg_site: SiteSnapshot, make_fetcher: Any) -> None:
    """Precision: the banner/footer documents and other categories are not returned."""
    result = _run(make_fetcher)
    extra = sorted({d.normalized_url for d in result.documents} - set(_ground_truth()))
    assert not extra
    assert len(result.documents) == 30


def test_metadata_matches_ground_truth(sg_site: SiteSnapshot, make_fetcher: Any) -> None:
    result = _run(make_fetcher)
    truth = _ground_truth()
    mismatches = []
    for doc in result.documents:
        expected = truth[doc.normalized_url]
        actual = {
            "type": doc.type,
            "period": doc.period,
            "fiscal_year": doc.fiscal_year,
            "language": doc.language,
            "date": doc.date,
            "title": doc.title,
        }
        wanted = {k: expected[k] for k in actual}
        if actual != wanted:
            mismatches.append((doc.url, actual, wanted))
    assert not mismatches


def test_pagination_follows_rel_next_only(sg_site: SiteSnapshot, make_fetcher: Any) -> None:
    result = _run(make_fetcher)
    assert [p.items for p in result.pages] == [10, 6, 10, 4]
    assert not sg_site.unexpected
    pages_requested = [u for u in sg_site.requested if "base-documentaire" in u]
    assert len(pages_requested) == 4
    assert not any("page=2" in u for u in pages_requested)


def test_robots_redirect_is_followed(sg_site: SiteSnapshot, make_fetcher: Any) -> None:
    fetcher = make_fetcher()
    run_recipe(load_recipe(RECIPE), fetcher, years=[2025], today=TODAY)
    assert fetcher.robots_outcome("https://investors.societegenerale.com/") == "parsed"


def test_two_runs_give_identical_results(sg_site: SiteSnapshot, make_fetcher: Any) -> None:
    first, second = _run(make_fetcher), _run(make_fetcher)
    assert first.documents == second.documents


@pytest.mark.parametrize("years", [[2024], [2025]])
def test_year_restriction(sg_site: SiteSnapshot, make_fetcher: Any, years: list[int]) -> None:
    result = run_recipe(load_recipe(RECIPE), make_fetcher(), years=years, today=TODAY)
    assert {d.fiscal_year for d in result.documents} == set(years)


def test_download_tree_manifest_and_dedup(
    sg_site: SiteSnapshot, make_fetcher: Any, tmp_path: Path
) -> None:
    recipe = load_recipe(RECIPE)
    fetcher = make_fetcher()
    result = run_recipe(recipe, fetcher, years=[2024, 2025], today=TODAY)
    downloader = Downloader(fetcher, tmp_path)

    outcomes = downloader.download_all(recipe, result.documents)
    assert [o.status for o in outcomes].count("downloaded") == 30, [
        (o.document.url, o.reason) for o in outcomes if o.status != "downloaded"
    ]
    manifest = Manifest.load(tmp_path)
    paths = [e.path for e in manifest.entries]
    assert len(set(paths)) == 30
    assert "societegenerale/2025/press_release_2025-Q4_fr.pdf" in paths
    assert "societegenerale/2024/quarterly_series_restated_2024-Q4_fr.xlsx" in paths
    for entry in manifest.entries:
        assert (tmp_path / entry.path).is_file()
        assert entry.recipe == "societegenerale-ir" and entry.recipe_version == 1

    again = Downloader(fetcher, tmp_path).download_all(recipe, result.documents)
    assert {o.status for o in again} == {"known_url"}
