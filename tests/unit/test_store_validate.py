"""Recipe store, SQLite state and validation gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrievr.classify import Decision
from retrievr.llm import CallRecord
from retrievr.recipe import FoundDocument, Recipe, RunResult, load_recipe
from retrievr.store import StateStore
from retrievr.validate import History, validate_runs

RECIPES = Path(__file__).resolve().parents[2] / "recipes"


@pytest.fixture
def recipe() -> Recipe:
    return load_recipe(RECIPES / "ubs-ir@1.yaml")


def _doc(url: str, type_: str | None = "press_release") -> FoundDocument:
    return FoundDocument(
        url=url,
        normalized_url=url,
        title="t",
        source_id="s",
        params={},
        page_url="https://www.ubs.com/",
        extension="pdf",
        date=None,
        type=type_,
        period="2025-Q1",
        fiscal_year=2025,
        language="en",
        score=1.0,
        justification="",
    )


def _run(*urls: str, status: str = "ok") -> RunResult:
    run = RunResult(recipe="ubs-ir", recipe_version=1, documents=[_doc(u) for u in urls])
    run.status = status  # type: ignore[assignment]
    return run


def test_recipes_are_immutable_and_activated(tmp_path: Path, recipe: Recipe) -> None:
    store = StateStore(tmp_path)
    assert store.next_version("ubs-ir") == 1
    path = store.save_recipe(recipe)
    assert store.next_version("ubs-ir") == 2
    with pytest.raises(FileExistsError):
        store.save_recipe(recipe)
    assert store.active_recipe("ubs") is None
    store.activate("ubs", recipe, path)
    assert store.active_recipe("ubs") == recipe  # YAML round trip keeps the recipe


def test_state_tables(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.save_decisions("r", {"https://a/x.pdf": Decision("urd", "2025-FY", 2025, "fr", "llm:t")})
    assert store.decisions("r")["https://a/x.pdf"].type == "urd"
    store.save_filter("h", "annual reports", '{"types": ["annual_report"]}')
    assert store.filter_json("h") == '{"types": ["annual_report"]}'
    assert store.filter_json("other") is None
    store.set_site_request("ubs", "annual reports")
    assert store.site_request("ubs") == "annual reports"
    run_id = store.start_run("learn", "ubs")
    record = CallRecord(
        "structure", "m", "k", [{"role": "user", "content": "x"}], "{}", 1, 2, 0.1, 0.5,
        False, "json_schema", 1, "2026-11-20T00:00:00+00:00",
    )  # fmt: skip
    store.record_llm_call(run_id, record)
    store.finish_run(run_id, status="learned", documents=3, llm_calls=1, cost_usd=0.1)
    (row,) = store.llm_calls(run_id)
    assert row["step"] == "structure" and row["cached"] == 0


def test_gates_without_history(recipe: Recipe) -> None:
    a = "https://www.ubs.com/a.pdf"
    report = validate_runs(recipe, _run(a), _run(a))
    assert report.passed, report.summary()
    assert not validate_runs(recipe, _run(), _run()).passed  # no document
    assert not validate_runs(recipe, _run(a), _run(a, "https://www.ubs.com/b.pdf")).passed
    outside = validate_runs(recipe, _run("https://evil.test/a.pdf"), None)
    assert [c.name for c in outside.failures()] == ["domains"]
    assert not validate_runs(recipe, _run(a, status="error"), None).passed


def test_gates_with_history(recipe: Recipe) -> None:
    urls = [f"https://www.ubs.com/{i}.pdf" for i in range(20)]
    history = History(anchors=urls[:2], known_online=urls, volume=20)
    assert validate_runs(recipe, _run(*urls), None, history=history).passed
    missing_anchor = validate_runs(recipe, _run(*urls[2:]), None, history=history)
    assert {c.name for c in missing_anchor.failures()} == {"anchors", "known_documents"}
    too_many = [*urls, *(f"https://www.ubs.com/x{i}.pdf" for i in range(11))]
    assert [
        c.name for c in validate_runs(recipe, _run(*too_many), None, history=history).failures()
    ] == ["volume"]
