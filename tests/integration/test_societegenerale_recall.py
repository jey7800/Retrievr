"""Site-specific behaviour of the Société Générale recipe (recall itself: test_recall.py)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from retrievr.download import Downloader, Manifest
from retrievr.recipe import load_recipe, run_recipe
from tests.conftest import RECIPES, SiteSnapshot

TODAY = date(2026, 11, 20)
RECIPE = RECIPES / "societegenerale-ir@1.yaml"


def test_pagination_follows_rel_next_only(sg_site: SiteSnapshot, make_fetcher: Any) -> None:
    result = run_recipe(load_recipe(RECIPE), make_fetcher(), years=[2024, 2025], today=TODAY)
    assert [p.items for p in result.pages] == [10, 6, 10, 4]
    assert not sg_site.unexpected
    pages_requested = [u for u in sg_site.requested if "base-documentaire" in u]
    assert len(pages_requested) == 4
    assert not any("page=2" in u for u in pages_requested)


def test_robots_redirect_is_followed(sg_site: SiteSnapshot, make_fetcher: Any) -> None:
    fetcher = make_fetcher()
    run_recipe(load_recipe(RECIPE), fetcher, years=[2025], today=TODAY)
    assert fetcher.robots_outcome("https://investors.societegenerale.com/") == "parsed"


@pytest.mark.parametrize("years", [[2024], [2025]])
def test_year_restriction(sg_site: SiteSnapshot, make_fetcher: Any, years: list[int]) -> None:
    result = run_recipe(load_recipe(RECIPE), make_fetcher(), years=years, today=TODAY)
    assert {d.fiscal_year for d in result.documents} == set(years)


def test_file_names(sg_site: SiteSnapshot, make_fetcher: Any, tmp_path: Path) -> None:
    recipe = load_recipe(RECIPE)
    fetcher = make_fetcher()
    result = run_recipe(recipe, fetcher, years=[2024, 2025], today=TODAY)
    Downloader(fetcher, tmp_path).download_all(recipe, result.documents)
    paths = {e.path for e in Manifest.load(tmp_path).entries}
    assert "societegenerale/2025/press_release_2025-Q4_fr.pdf" in paths
    assert "societegenerale/2024/quarterly_series_restated_2024-Q4_fr.xlsx" in paths
