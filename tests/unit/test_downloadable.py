"""Learning checks that the listed documents can be downloaded: a listing API whose items
carry a storage path (404 for the public) and a public link must use the public link."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from retrievr.api import Retrievr
from retrievr.config import BrowserSettings, Settings, StateSettings
from retrievr.recipe import JsonItems, RequestSource, recipe_from_mapping, run_recipe
from retrievr.snapshots import synthetic_document
from retrievr.validate import Reachability, validate_runs

SITE = "https://ir.example-issuer.com"
TODAY = date(2026, 11, 23)


def _items() -> list[dict[str, str]]:
    return [
        {
            "uri": f"{SITE}/sites/default/files/{y}/annual-report-{y}.pdf",
            "url": f"{SITE}/en/doc/annual-report-{y}.pdf",
            "name": f"Annual report {y}",
        }
        for y in (2023, 2024, 2025)
    ]


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/documents":
        return httpx.Response(200, json={"results": _items()})
    if path.startswith("/en/doc/"):  # the public links serve the files
        return httpx.Response(200, content=synthetic_document("pdf", str(request.url)) or b"")
    return httpx.Response(404)  # robots.txt, and the storage paths


def _recipe(paths: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "recipe": "example-ir",
        "version": 1,
        "issuer": "example-issuer",
        "site": "ir.example-issuer.com",
        "allowed_domains": ["ir.example-issuer.com"],
        "sources": [
            {
                "id": "documents",
                "kind": "request",
                "method": "GET",
                "url": f"{SITE}/api/documents",
                "response": {
                    "format": "json",
                    "items": {"path": "results", "url": paths, "title": "name"},
                },
                "items": {"extensions": ["pdf"]},
            }
        ],
    }


def _retrievr(tmp_path: Path) -> Retrievr:
    return Retrievr(
        settings=Settings(
            state=StateSettings(state_dir=tmp_path / "state"),
            browser=BrowserSettings(enabled=False),
        ),
        fetch_transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        today=TODAY,
    )


def test_the_public_link_is_preferred_to_a_storage_path_that_answers_404(
    tmp_path: Path,
) -> None:
    r = _retrievr(tmp_path)
    recipe = recipe_from_mapping(_recipe(["uri", "url"]))
    with r._fetcher() as fetcher:
        first = run_recipe(recipe, fetcher, today=TODAY)
        assert all("/sites/default/files/" in d.url for d in first.documents)
        chosen, run, reach = r._downloadable(recipe, first, fetcher, None)
    source = chosen.sources[0]
    assert isinstance(source, RequestSource) and source.response is not None
    assert isinstance(source.response.items, JsonItems)
    assert source.response.items.url == ["url", "uri"]
    assert all("/en/doc/" in d.url for d in run.documents)
    assert reach == Reachability(sampled=3, reachable=3, gone=0)


def test_a_recipe_whose_documents_all_answer_404_is_not_activated(tmp_path: Path) -> None:
    r = _retrievr(tmp_path)
    recipe = recipe_from_mapping(_recipe(["uri"]))
    with r._fetcher() as fetcher:
        first = run_recipe(recipe, fetcher, today=TODAY)
        _chosen, run, reach = r._downloadable(recipe, first, fetcher, None)
    assert reach == Reachability(sampled=3, reachable=0, gone=3)
    report = validate_runs(recipe, run, run, reachability=reach)
    assert not report.passed
    assert "downloadable" in report.summary()


def test_a_few_removed_documents_do_not_reject_the_recipe() -> None:
    assert Reachability(sampled=3, reachable=1, gone=2).ok
    assert Reachability(sampled=3, reachable=0, gone=0).ok  # blocked or network: unknown
    assert not Reachability(sampled=2, reachable=0, gone=2).ok
