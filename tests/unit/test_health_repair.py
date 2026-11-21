"""Health verdicts, history state and the repair ladder's limits (no network, no LLM)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from retrievr.config import FetchSettings
from retrievr.fetch import Fetcher
from retrievr.health import HealthReport, KnownDocument, assess, period_end, published
from retrievr.recipe import FoundDocument, PageVisit, Recipe, RunResult, load_recipe
from retrievr.repair import (
    RepairContext,
    Repairer,
    _compound,
    _next_page_of,
    _segment,
)
from retrievr.store import StateStore

RECIPES = Path(__file__).resolve().parents[2] / "recipes"
TODAY = date(2026, 11, 20)
BASE = "https://investors.societegenerale.com/fr/base-documentaire"


@pytest.fixture
def recipe() -> Recipe:
    return load_recipe(RECIPES / "societegenerale-ir@1.yaml")


def _doc(i: int, *, type_: str = "press_release", year: int = 2025) -> FoundDocument:
    url = f"https://www.societegenerale.com/files/doc-{year}-{i}.pdf"
    return FoundDocument(
        url=url,
        normalized_url=url,
        title=f"Document {i}",
        source_id="resultats-financiers",
        params={"year": year},
        page_url=BASE,
        extension="pdf",
        date=None,
        type=type_,
        period=f"{year}-Q{1 + i % 4}",
        fiscal_year=year,
        language="fr",
        score=1.0,
        justification="",
    )


def _docs(n: int = 30) -> list[FoundDocument]:
    return [_doc(i, year=2024 + i % 2) for i in range(n)]


def _run(recipe: Recipe, docs: list[FoundDocument], **kwargs: Any) -> RunResult:
    run = RunResult(recipe=recipe.recipe, recipe_version=recipe.version, documents=docs, **kwargs)
    run.pages.append(PageVisit("resultats-financiers", BASE, BASE, 200, len(docs)))
    return run


def _assess(
    recipe: Recipe, now: list[FoundDocument], before: list[FoundDocument], **kw: Any
) -> HealthReport:
    known = [KnownDocument.of(d) for d in before]
    return assess(
        recipe, _run(recipe, now), now, known, site="sg", today=TODAY, years=[2024, 2025], **kw
    )


def _without_anchors(recipe: Recipe) -> Recipe:
    return recipe.model_copy(update={"health": recipe.health.model_copy(update={"anchors": []})})


def test_anchors_are_checked_on_unrestricted_runs(recipe: Recipe) -> None:
    report = assess(recipe, _run(recipe, _docs()), _docs(), [], site="sg", today=TODAY)
    assert report.verdict == "broken"  # the recipe's anchors are not listed


def test_first_run_is_a_baseline(recipe: Recipe) -> None:
    report = _assess(recipe, _docs(), [])
    assert report.verdict == "ok" and report.baseline


def test_unchanged_listing_is_ok(recipe: Recipe) -> None:
    assert _assess(recipe, _docs(), _docs()).verdict == "ok"


def test_one_missing_document_is_never_ok(recipe: Recipe) -> None:
    report = _assess(recipe, _docs()[1:], _docs())
    assert report.verdict == "suspect"
    assert report.missing == [_docs()[0].normalized_url]
    assert not report.repair_needed


def test_many_missing_documents_are_broken(recipe: Recipe) -> None:
    report = _assess(recipe, _docs()[5:], _docs())
    assert report.verdict == "broken" and report.repair_needed


def test_documents_gone_from_the_site_are_not_a_breakage(recipe: Recipe) -> None:
    report = _assess(recipe, _docs()[2:], _docs(), probe=lambda _url: False)
    assert report.verdict == "ok"
    assert len(report.removed) == 2 and not report.missing


def test_missing_documents_still_online_count(recipe: Recipe) -> None:
    report = _assess(recipe, _docs()[5:], _docs(), probe=lambda _url: True)
    assert report.verdict == "broken"


def test_changed_metadata_is_detected(recipe: Recipe) -> None:
    now = _docs()
    now[0] = _doc(0, type_="results_presentation", year=2024)
    assert _assess(recipe, now, _docs()).verdict == "suspect"
    now = [_doc(i, type_="results_presentation", year=2024 + i % 2) for i in range(30)]
    report = _assess(recipe, now, _docs())
    assert report.verdict == "broken" and len(report.drifted) == 30


def test_no_document_is_broken_and_a_block_is_blocked(recipe: Recipe) -> None:
    assert _assess(recipe, [], _docs()).verdict == "broken"
    blocked = _run(recipe, [], status="blocked", errors=["blocked: HTTP 403"])
    report = assess(recipe, blocked, [], [], site="sg", today=TODAY)
    assert report.verdict == "blocked" and not report.repair_needed


def test_run_errors_404_is_broken_timeouts_are_suspect(recipe: Recipe) -> None:
    recipe = _without_anchors(recipe)
    gone = _run(recipe, _docs(), status="error", errors=["s: HTTP 404 at https://x/"])
    assert assess(recipe, gone, _docs(), [], site="sg", today=TODAY).verdict == "broken"
    slow = _run(recipe, _docs(), status="error", errors=["s: network error at https://x/"])
    assert assess(recipe, slow, _docs(), [], site="sg", today=TODAY).verdict == "suspect"


def test_too_many_documents_are_suspect(recipe: Recipe) -> None:
    extra = [_doc(100 + i, year=2025) for i in range(20)]
    report = _assess(recipe, _docs() + extra, _docs())
    assert report.verdict == "suspect"


def test_stale_when_nothing_was_published_for_too_long(recipe: Recipe) -> None:
    recipe = _without_anchors(recipe)
    old = [_doc(i, year=2024) for i in range(10)]
    report = assess(recipe, _run(recipe, old), old, [], site="sg", today=TODAY)
    assert report.verdict == "stale"


def test_publication_estimate() -> None:
    assert period_end("2025-Q3", 2025) == date(2025, 9, 30)
    assert period_end("2025-H1", 2025) == date(2025, 6, 30)
    assert period_end(None, 2024) == date(2024, 12, 31)
    assert published(_doc(2, year=2025)) == date(2025, 11, 29)  # Q3 end + 60 days


def test_history_is_stored_per_site(tmp_path: Path, recipe: Recipe) -> None:
    store = StateStore(tmp_path)
    store.record_documents("sg", recipe.key, _docs(4))
    store.record_documents("sg", recipe.key, [_doc(0, type_="urd", year=2024)])
    known = store.known_documents("sg")
    assert len(known) == 4 and known[0].params == {"year": 2024}
    assert {k.type for k in known} == {"urd", "press_release"}
    store.mark_removed("sg", [known[0].normalized_url])
    assert len(store.known_documents("sg")) == 3
    assert store.known_documents("other") == []


def test_daily_limit_of_llm_repairs(tmp_path: Path, recipe: Recipe) -> None:
    store = StateStore(tmp_path)
    for rung in ("relocate", "llm:small", "llm:large"):
        store.record_repair(
            "sg", None, TODAY, rung=rung, outcome="failed", recipe_key=None, cost_usd=0, detail=""
        )
    assert store.llm_repairs_on("sg", TODAY) == 2
    assert store.llm_repairs_on("sg", date(2026, 11, 21)) == 0


class _NeverCalled:
    def structured(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the LLM must not be called")


def test_repair_without_llm_attempts_left_ends_degraded(recipe: Recipe) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("robots.txt"):
            return httpx.Response(404)
        return httpx.Response(200, text="<html><body><p>Nothing here</p></body></html>")

    known = [KnownDocument.of(d) for d in _docs()]
    report = _assess(recipe, [], _docs())
    ctx = RepairContext(
        site="sg",
        today=TODAY,
        years=[2024, 2025],
        version=2,
        llm=_NeverCalled(),
        llm_attempts_left=0,
    )
    with Fetcher(
        FetchSettings(), transport=httpx.MockTransport(handler), sleep=lambda _s: None
    ) as f:
        result = Repairer(recipe, report, known, f, ctx).repair()
    assert result.status == "degraded"
    assert result.reason == "daily limit of LLM repair attempts reached"
    assert result.recipe is None


def test_repair_needs_known_documents(recipe: Recipe) -> None:
    report = _assess(recipe, [], [])
    ctx = RepairContext(site="sg", today=TODAY, years=None, version=2)
    with Fetcher(
        FetchSettings(), transport=httpx.MockTransport(lambda r: httpx.Response(404))
    ) as f:
        result = Repairer(recipe, report, [], f, ctx).repair()
    assert result.status == "degraded" and "no known document" in result.reason


def test_selector_helpers() -> None:
    assert _next_page_of(f"{BASE}?fiscal%5B2024%5D=", f"{BASE}?fiscal[2024]=&page=1")
    assert _next_page_of(f"{BASE}?a=1&page=2", f"{BASE}?a=1&page=3")
    assert not _next_page_of(f"{BASE}?a=1&page=2", f"{BASE}?a=2&page=3")
    assert _segment("category%5Bx%5D=") == _segment("category[x]")
    from retrievr.candidates import parse_html

    root = parse_html(
        '<ul><li class="doc card x123456"><a href="a.pdf">a</a></li>'
        '<li class="doc"><a href="b.pdf">b</a></li></ul>'
    )
    items = list(root.iter("li"))
    assert _compound(items) == "li.doc"
